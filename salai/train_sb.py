import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn

import speechbrain as sb
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.encoder import CTCTextEncoder
from speechbrain.decoders.ctc import CTCBeamSearcher, ctc_greedy_decode
from hyperpyyaml import load_hyperpyyaml

try:
    from .sb_utils import build_model_from_hparams
    from ..common.text_norm import normalize_atc_text
    from ..common.metrics import compute_wer
except Exception:
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from asr_finetune.speechbrain.sb_utils import (
    build_model_from_hparams,
    inject_lora_adapters,
    save_model_card,
    )
    from asr_finetune.common.text_norm import normalize_atc_text
    from asr_finetune.common.metrics import compute_wer


@dataclass
class SBArgs:
    # Data
    train_manifest: str
    valid_manifest: str
    test_manifest: Optional[str]
    output_dir: str
    dataset_part: str
    train_fraction: float
    validation_samples: int
    train_subset: int
    skip_test: bool
    # Optimization
    epochs: int
    lr: float
    batch_size: int
    grad_accum: int
    seed: int
    # Model
    model_name: str
    hparams_file: str
    # Tokenizer
    tokenizer_type: str  # 'char' or 'spm'
    spm_model: Optional[str]
    # PEFT
    peft: str  # 'none' or 'lora'
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    # Decode
    beam_size: int
    lm_weight: float
    # Logging / meta
    run_id: Optional[str]
    run_stage: Optional[str]
    metrics_csv: Optional[str]
    notes: str
    save_full_model: bool
    language: str
    task: str
    eval_strategy: str
    eval_steps: Optional[int]


class ASRBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        feats = self.hparams.compute_features(wavs)
        feats = self.hparams.normalize(feats, wav_lens)
        if self.hparams.specaug is not None and stage == sb.Stage.TRAIN:
            feats = self.hparams.specaug(feats, wav_lens)
        enc_out = self.modules.encoder(feats)
        if isinstance(enc_out, tuple):
            enc_out = enc_out[0]
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)
        return p_ctc, wav_lens

    def _indices_to_text(self, idx_seqs: List[List[int]]) -> List[str]:
        # Convert index sequences to text using encoder's symbol list
        vocab = self.hparams.label_encoder.ind2lab
        return ["".join(vocab[i] for i in seq) for seq in idx_seqs]

    def compute_objectives(self, predictions, batch, stage):
        p_ctc, wav_lens = predictions
        tokens, tokens_lens = batch.tokens, batch.tokens_lens
        ctc_loss = self.hparams.ctc_cost(
            p_ctc,
            tokens,
            wav_lens,
            tokens_lens,
        )
        if stage != sb.Stage.TRAIN:
            if self.hparams.val_beam_size and self.hparams.val_beam_size > 1:
                beams_per_utt = self.hparams.ctc_beam_searcher(p_ctc, wav_lens)
                hyps = [beams[0].text for beams in beams_per_utt]  # top-1 text
            else:
                # Greedy with proper CTC collapsing
                idxs = ctc_greedy_decode(p_ctc, wav_lens, blank_id=self.hparams.label_encoder.get_blank_index())
                hyps = self._indices_to_text(idxs)

            # Build references as strings
            refs = []
            vocab = self.hparams.label_encoder.ind2lab
            for t, l in zip(batch.tokens, batch.tokens_lens):
                ids = t[: int(l)].tolist()
                refs.append("".join(vocab[i] for i in ids))

            self.wer_metric.append(batch.id, hyps, refs)
        return ctc_loss

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.Stage.TRAIN:
            self.wer_metric = sb.utils.metric_stats.ErrorRateStats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.hparams._last_train_loss = float(stage_loss)
        if stage == sb.Stage.VALID:
            self.hparams._last_valid_loss = float(stage_loss)
        return super().on_stage_end(stage, stage_loss, epoch)


def _read_manifest(path):
    items = []
    if path.endswith(".json") or path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
    else:
        import pandas as pd
        df = pd.read_csv(path)
        items = df.to_dict("records")
    return items


def _to_dataset(items, tokenizer):
    # Convert list[dict] -> dict[id] = {wav, transcript, ...}  (no 'id' field inside)
    data = {}
    for ex in items:
        ex_id = str(ex["id"])
        data[ex_id] = {k: v for k, v in ex.items() if k != "id"}

    ds = DynamicItemDataset(data)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig", "sig_len")
    def audio_pipeline(wav_path):
        import torchaudio
        sig, sr = torchaudio.load(wav_path)
        if sr != 16000:
            sig = torchaudio.functional.resample(sig, sr, 16000)
            sr = 16000
        if sig.shape[0] > 1:
            sig = torch.mean(sig, dim=0, keepdim=True)
        sig = sig.squeeze(0)
        return sig, torch.tensor([sig.shape[-1] / sr], dtype=torch.float32)

    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides("tokens", "tokens_lens")
    def text_pipeline(text):
        norm = normalize_atc_text(text)
        ids = tokenizer.encode_sequence(norm)
        encoded = torch.tensor(ids, dtype=torch.long)
        return encoded, torch.tensor([len(ids)], dtype=torch.long)

    ds.add_dynamic_item(audio_pipeline)
    ds.add_dynamic_item(text_pipeline)
    # 'id' is the reserved key; you can request it as an output
    ds.set_output_keys(["id", "sig", "tokens", "tokens_lens"])
    return ds

def load_datasets(train_path: str, valid_path: str, test_path: Optional[str], tokenizer: CTCTextEncoder, train_subset: int = 0):
    train_items = _read_manifest(train_path)
    valid_items = _read_manifest(valid_path)
    test_items = _read_manifest(test_path) if test_path else None

    if train_subset and train_subset > 0:
        train_items = train_items[:train_subset]

    train_ds = _to_dataset(train_items, tokenizer)
    valid_ds = _to_dataset(valid_items, tokenizer)
    test_ds = _to_dataset(test_items, tokenizer) if test_items else None
    return train_ds, valid_ds, test_ds


def make_tokenizer(tokenizer_type: str, spm_model: Optional[str]) -> CTCTextEncoder:
    enc = CTCTextEncoder()
    if tokenizer_type == "char":
        symbols = [" "] + [chr(c) for c in range(65, 91)] + [str(d) for d in range(10)] + [".", "/"]
        enc.update_from_iterable(symbols)
        enc.insert_blank(index=0)  # <-- critical for CTC
    elif tokenizer_type == "spm":
        if not spm_model:
            raise ValueError("SentencePiece model path required for tokenizer_type=spm")
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=spm_model)
        for i in range(sp.get_piece_size()):
            enc.update_from_iterable([sp.id_to_piece(i)])
        enc.sp_model = sp
        enc.insert_blank(index=0)
    else:
        raise ValueError("tokenizer_type must be 'char' or 'spm'")
    return enc


def main():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--train_manifest", required=True)
    p.add_argument("--valid_manifest", required=True)
    p.add_argument("--test_manifest", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset_part", default="base")
    p.add_argument("--train_fraction", type=float, default=1.0)
    p.add_argument("--validation_samples", type=int, default=0)
    p.add_argument("--train_subset", type=int, default=0)
    p.add_argument("--skip_test", action="store_true")
    # Optimization
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    # Model
    p.add_argument("--model_name", default="sb-conformer-ctc")
    p.add_argument("--hparams_file", default=os.path.join(os.path.dirname(__file__), "hparams_sb.yaml"))
    # Tokenizer
    p.add_argument("--tokenizer_type", choices=["char", "spm"], default="char")
    p.add_argument("--spm_model", default=None)
    # PEFT
    p.add_argument("--peft", choices=["none", "lora"], default="none")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # Decode
    p.add_argument("--beam_size", type=int, default=8)
    p.add_argument("--lm_weight", type=float, default=0.0)
    # Logging/meta
    p.add_argument("--run_id", default=None)
    p.add_argument("--run_stage", default="single")
    p.add_argument("--metrics_csv", default=None)
    p.add_argument("--notes", default="")
    p.add_argument("--save_full_model", action="store_true")
    p.add_argument("--language", default="en")
    p.add_argument("--task", default="transcribe")
    p.add_argument("--eval_strategy", default="epoch")
    p.add_argument("--eval_steps", type=int, default=None)

    args = p.parse_args()
    torch.manual_seed(args.seed)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or ts

    os.makedirs(args.output_dir, exist_ok=True)
    run_dir = os.path.join(args.output_dir, f"{args.model_name}_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # Load YAML hparams
    with open(args.hparams_file, "r", encoding="utf-8") as f:
        hparams = load_hyperpyyaml(f)

    # Tokenizer
    label_encoder = make_tokenizer(args.tokenizer_type, args.spm_model)

    space_idx = label_encoder.ind2lab.index(
        " ") if " " in label_encoder.ind2lab else -1
    hparams["ctc_beam_searcher"] = CTCBeamSearcher(
        blank_index=label_encoder.get_blank_index(),
        vocab_list=label_encoder.ind2lab,
        space_token=space_idx,
        beam_size=args.beam_size,
    )

    # Datasets
    train_ds, valid_ds, test_ds = load_datasets(args.train_manifest, args.valid_manifest, args.test_manifest, label_encoder, args.train_subset)

    # Feature pipeline and model
    modules, hparams = build_model_from_hparams(hparams, label_encoder)

    # Build beam searcher _after_ we know the vocab
    space_index = label_encoder.ind2lab.index(" ") if " " in label_encoder.ind2lab else -1
    hparams["ctc_beam_searcher"] = CTCBeamSearcher(
        blank_index=label_encoder.get_blank_index(),
        vocab_list=label_encoder.ind2lab,
        space_token=space_index,
        beam_size=args.beam_size,
    )

    # LoRA injection if requested
    if args.peft == "lora":
        inject_lora_adapters(modules, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

    # Brain
    train_logger = sb.utils.train_logger.FileTrainLogger(save_file=os.path.join(run_dir, "train_log.txt"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None

    brain = ASRBrain(modules=modules,
                     opt_class=lambda x: torch.optim.AdamW(x, lr=args.lr),
                     hparams={**hparams, "label_encoder": label_encoder, "train_logger": train_logger,
                              "val_beam_size": args.beam_size, "_last_train_loss": None, "_last_valid_loss": None},
                     run_opts={"device": device, "auto_mix_prec": True},
                     checkpointer=sb.utils.checkpoints.Checkpointer(run_dir, recoverables=modules))

    # Samplers and loaders (dynamic length-based batching)
    train_sampler = DynamicBatchSampler(train_ds, dynamic_items=["sig"], max_batch_len=args.batch_size * 16000 * 10)
    train_loader = sb.dataio.dataloader.SaveableDataLoader(train_ds, batch_size=None, sampler=train_sampler)
    valid_loader = sb.dataio.dataloader.SaveableDataLoader(valid_ds, batch_size=1)

    # Train
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    brain.fit(epoch_counter=range(1, args.epochs + 1),
              train_set=train_loader,
              valid_set=valid_loader,
              train_loader_kwargs={"drop_last": True},
              valid_loader_kwargs={})
    train_time = time.time() - t0

    # Save artifacts
    brain.checkpointer.save_checkpoint(meta={"run_dir": run_dir})
    save_model_card(run_dir, args, hparams)

    # Evaluate on valid and test
    t_inf0 = time.time()
    valid_wer = sb.utils.data_utils.to_float(compute_wer(brain, valid_ds, label_encoder, beam_size=args.beam_size))
    inference_time = time.time() - t_inf0

    test_wer = None
    if (args.test_manifest is not None) and (not args.skip_test):
        test_wer = sb.utils.data_utils.to_float(compute_wer(brain, test_ds, label_encoder, beam_size=args.beam_size))

    peak_mem_mb = None
    if torch.cuda.is_available():
        peak_mem_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)

    # Build sweep-compatible record
    record = {
        "timestamp": ts,
        "run_id": run_id,
        "run_stage": args.run_stage,
        "model": args.model_name,
        "output_dir": run_dir,
        "dataset_part": args.dataset_part,
        "train_fraction": args.train_fraction,
        "validation_samples": args.validation_samples,
        "epochs": args.epochs,
        "wer": valid_wer,
        "train_runtime": round(train_time, 2),
        "train_loss": brain.hparams._last_train_loss,
        "eval_loss": brain.hparams._last_valid_loss,
        "device": device,
        "gpu": gpu_name,
        "per_device_train_batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lora_r": args.lora_r if args.peft == "lora" else None,
        "lora_alpha": args.lora_alpha if args.peft == "lora" else None,
        "lora_dropout": args.lora_dropout if args.peft == "lora" else None,
        "quant_weights_dtype": "fp32",
        "quant_compute_dtype": "float16" if device == "cuda" else "float32",
        "inference_time_sec": round(inference_time, 2),
        "peak_memory_mb": peak_mem_mb,
        "save_full_model": bool(args.save_full_model),
        "language": args.language,
        "task": args.task,
        "learning_rate": args.lr,
        "seed": args.seed,
        "eval_strategy": args.eval_strategy,
        "eval_steps": args.eval_steps,
    }

    # Append to CSV if requested
    if args.metrics_csv:
        write_header = not os.path.isfile(args.metrics_csv)
        with open(args.metrics_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(record)

    # Always dump JSON in run dir
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as jf:
        json.dump(record, jf, indent=2)


if __name__ == "__main__":
    main()
