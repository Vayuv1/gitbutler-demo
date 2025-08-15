# salai/asr_finetune/speechbrain/train_sb.py
"""
This script is the main entry point for training a SpeechBrain ASR model.
It handles everything from argument parsing and data loading to training,
validation, and artifact generation.
"""

import argparse
import os
import json
import time
import torch
import logging
import pandas as pd
import sys
from datetime import datetime
from pathlib import Path

import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.sampler import DynamicBatchSampler
from torch.utils.data import DataLoader
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.metric_stats import ErrorRateStats
from torch.optim import AdamW
from speechbrain.utils.seed import seed_everything

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from asr_finetune.speechbrain.sb_utils import (
    build_asr_model,
    save_model_card,
    inject_lora,
    create_char_tokenizer,
)
from asr_finetune.common.text_norm import normalize_transcript
from asr_finetune.common.logging_utils import (
    get_git_commit_hash,
    log_gpu_memory,
    get_peak_memory_mb,
)

logger = logging.getLogger(__name__)


class ASR(sb.Brain):
    """
    Main class for ASR training and evaluation.
    """

    def compute_forward(self, batch, stage):
        """
        Forward pass of the model.
        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        feats = self.hparams.compute_features(wavs)

        # Augmentation
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "augmenter"):
            feats, wav_lens = self.hparams.augmenter(feats, wav_lens)

        # Project features to model dimension
        feats = self.modules.input_linear(feats)

        # Model forward
        out, _ = self.modules.model(feats)  # ConformerEncoder returns (x, attn)
        logits = self.modules.ctc_lin(out)
        p_ctc = self.hparams.log_softmax(logits)

        return p_ctc, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """
        Computes the loss.
        """
        p_ctc, wav_lens = predictions
        ids = batch.id
        tokens, tokens_lens = batch.tokens

        loss = self.hparams.ctc_loss(p_ctc, tokens, wav_lens, tokens_lens)

        if stage != sb.Stage.TRAIN:
            # Decode the outputs for WER/CER calculation
            predicted_tokens = p_ctc.argmax(dim=-1, keepdim=False)
            predicted_words = self.hparams.tokenizer.decode_ndim(
                predicted_tokens)

            target_words = batch.words
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        """
        Overrides fit_batch to implement gradient accumulation and step-based evaluation.
        """
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            with torch.autocast(torch.device(self.device).type):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            self.scaler.scale(loss / self.hparams.grad_accum_steps).backward()
            if self.step % self.hparams.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                if self.check_gradients():
                    self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            (loss / self.hparams.grad_accum_steps).backward()
            if self.step % self.hparams.grad_accum_steps == 0:
                if self.check_gradients():
                    self.optimizer.step()
                self.optimizer.zero_grad()

        # Step-based evaluation
        if self.hparams.eval_strategy == "steps" and self.step % self.hparams.eval_steps == 0 and self.step > 0:
            logger.info(f"Running validation at step {self.step}")
            self.evaluate(self.hparams.valid_set, min_key="WER")

        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        """
        Gets called at the beginning of each epoch.
        """
        self.wer_metric = ErrorRateStats()
        self.cer_metric = ErrorRateStats(split_tokens=True)

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """
        Gets called at the end of an epoch.
        """
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        elif stage == sb.Stage.VALID:
            wer = self.wer_metric.summarize("error_rate")
            cer = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = wer
            stage_stats["CER"] = cer

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "step": self.step},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": wer}, min_keys=["WER"]
            )
            self.valid_stats = stage_stats  # Save for final logging

        elif stage == sb.Stage.TEST:
            wer = self.wer_metric.summarize("error_rate")
            cer = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = wer
            stage_stats["CER"] = cer

            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if hasattr(self.hparams, "wer_file") and self.hparams.wer_file:
                with open(self.hparams.wer_file, "w") as w:
                    self.wer_metric.write_stats(w)
            self.test_stats = stage_stats


def dataio_prepare(hparams, tokenizer):
    """
    Prepares the data for training.
    """

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        return sb.dataio.dataio.read_audio(wav)

    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides("words", "tokens")
    def text_pipeline(transcript):
        words = normalize_transcript(transcript)
        tokens_list = tokenizer.encode_sequence(words)
        tokens = torch.LongTensor(tokens_list)
        return words, tokens

    datasets = {}
    manifests = {
        "train": hparams["train_manifest"],
        "valid": hparams["valid_manifest"],
        "test": hparams.get("test_manifest")
    }

    for key, manifest_path in manifests.items():
        if manifest_path:
            datasets[key] = DynamicItemDataset.from_csv(
                csv_path=manifest_path,
                replacements={"data_root": ""},
            )
            datasets[key].add_dynamic_item(audio_pipeline)
            datasets[key].add_dynamic_item(text_pipeline)
            datasets[key].set_output_keys(
                ["id", "sig", "words", "tokens", "duration"])
        else:
            datasets[key] = None

    if hparams.get("train_subset") and datasets["train"]:
        datasets["train"] = datasets["train"].filtered_sorted(
            sort_key="duration",
            select_n=hparams["train_subset"]
        )

    return datasets["train"], datasets["valid"], datasets["test"]


def setup_logging(log_file):
    """Configure logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="SpeechBrain ASR Finetuning")

    # Data args
    parser.add_argument("--train_manifest", type=str, required=True,
                        help="Path to the training manifest CSV.")
    parser.add_argument("--valid_manifest", type=str, required=True,
                        help="Path to the validation manifest CSV.")
    parser.add_argument("--test_manifest", type=str, default=None,
                        help="Path to the test manifest CSV.")
    parser.add_argument("--dataset_part", type=str, required=True,
                        help="Name of the dataset part (e.g., 'base', 'part2').")
    parser.add_argument("--train_subset", type=int, default=None,
                        help="Use a small subset of the training data for quick runs.")
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip the final test set evaluation.")

    # Optimization args
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Effective batch size.")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Model args
    parser.add_argument("--model_name", type=str, default="conformer-ctc",
                        help="Name of the model for saving.")
    parser.add_argument("--hparams_file", type=str,
                        default="asr_finetune/speechbrain/hparams_sb.yaml",
                        help="Path to the hyperparameters file.")

    # Tokenizer args
    parser.add_argument("--tokenizer_type", type=str, default="char",
                        choices=["char", "spm"],
                        help="Type of tokenizer to use.")
    parser.add_argument("--spm_model", type=str, default=None,
                        help="Path to SentencePiece model file (if tokenizer_type is 'spm').")

    # PEFT args
    parser.add_argument("--peft", type=str, default="none",
                        choices=["none", "lora"],
                        help="Parameter-Efficient Finetuning method.")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout.")

    # Logging & meta args
    parser.add_argument("--run_id", type=str,
                        default=f"sb_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        help="Unique ID for the run.")
    parser.add_argument("--run_stage", type=str, default="train",
                        help="Stage of the run (e.g., 'train', 'test').")
    parser.add_argument("--metrics_csv", type=str, default=None,
                        help="Path to master CSV for logging sweep results.")
    parser.add_argument("--notes", type=str, default="",
                        help="Optional notes for the run.")
    parser.add_argument("--save_full_model", action="store_true",
                        help="Save the full model state_dict, not just the checkpointer.")
    parser.add_argument("--language", type=str, default="en",
                        help="Language of the dataset.")
    parser.add_argument("--task", type=str, default="asr", help="Task type.")
    parser.add_argument("--eval_strategy", type=str, default="epoch",
                        choices=["epoch", "steps"], help="Evaluation strategy.")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Evaluation frequency in steps (if eval_strategy is 'steps').")

    args = parser.parse_args()

    # Define output directory and overrides
    output_dir = Path(
        f"finetuned_models/speechbrain/{args.model_name}_{args.run_id}")
    overrides = {"output_folder": str(output_dir)}

    # Create experiment directory, passing overrides to resolve YAML references
    sb.create_experiment_directory(
        experiment_directory=output_dir,
        hyperparams_to_save=args.hparams_file,
        overrides=overrides,
    )
    setup_logging(output_dir / "train_log.txt")

    # Load hyperparameters with overrides
    with open(args.hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Override hparams with command-line arguments
    hparams.update(vars(args))

    # Set seed for reproducibility
    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True

    # Create tokenizer
    if args.tokenizer_type == "char":
        tokenizer = create_char_tokenizer()
        hparams["tokenizer"] = tokenizer
        vocab_size = len(tokenizer)
        hparams["output_neurons"] = vocab_size
        tokenizer.expect_len(vocab_size)
    elif args.tokenizer_type == "spm":
        if not args.spm_model:
            raise ValueError("--spm_model is required for tokenizer_type 'spm'")
        hparams["tokenizer"] = SentencePiece(
            model_dir=Path(args.spm_model).parent,
            vocab_size=hparams["output_neurons"])
    else:
        raise ValueError(f"Unsupported tokenizer_type: {args.tokenizer_type}")

    # Data IO
    train_data, valid_data, test_data = dataio_prepare(hparams,
                                                       hparams["tokenizer"])
    hparams[
        "valid_set"] = valid_data  # Pass validation set to Brain for step-based eval

    # Model and Brain
    model = build_asr_model(hparams)
    if args.peft == "lora":
        model = inject_lora(model, r=args.lora_r, alpha=args.lora_alpha,
                            dropout=args.lora_dropout)

    # Manually update the epoch counter limit from the parsed arguments
    hparams["epoch_counter"].limit = args.epochs

    asr_brain = ASR(
        modules={
            "model": model,
            "ctc_lin": hparams["ctc_lin"],
            "input_linear": hparams["input_linear"]
        },
        opt_class=lambda params: AdamW(params, lr=args.lr),
        hparams=hparams,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        checkpointer=hparams["checkpointer"],
    )

    # Pass grad_accum to Brain
    asr_brain.hparams.grad_accum_steps = args.grad_accum

    # Determine validation set for fit method
    fit_valid_set = None if args.eval_strategy == "steps" else valid_data

    # Training
    logger.info(f"Starting training run: {args.run_id}")
    start_time = time.time()
    asr_brain.fit(
        epoch_counter=asr_brain.hparams.epoch_counter,
        train_set=train_data,
        valid_set=fit_valid_set,
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )
    train_runtime = time.time() - start_time

    # Evaluation
    test_wer = -1
    if not args.skip_test and test_data:
        logger.info("Running final evaluation on the test set...")
        asr_brain.hparams.wer_file = output_dir / "wer_test_stats.txt"
        asr_brain.evaluate(
            test_data,
            min_key="WER",
            test_loader_kwargs=hparams["dataloader_opts"],
        )
        test_wer = asr_brain.test_stats["WER"]

    # Save artifacts
    if args.save_full_model:
        torch.save(model.state_dict(), output_dir / "model_state.pt")

    # Save tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    hparams["tokenizer"].save(tokenizer_path)
    logger.info(f"Tokenizer saved to {tokenizer_path}")

    save_model_card(output_dir, hparams)

    # Log metrics
    peak_memory = get_peak_memory_mb()
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "run_id": args.run_id,
        "run_stage": args.run_stage,
        "model": args.model_name,
        "output_dir": str(output_dir),
        "dataset_part": args.dataset_part,
        "train_fraction": 1.0,
        "validation_samples": len(valid_data) if valid_data else 0,
        "epochs": args.epochs,
        "wer": test_wer,
        "train_runtime": train_runtime,
        "train_loss": asr_brain.train_stats["loss"],
        "eval_loss": asr_brain.valid_stats["loss"],
        "device": torch.cuda.get_device_name(
            0) if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.device_count(),
        "per_device_train_batch_size": hparams["dataloader_opts"]["batch_size"],
        "grad_accum": args.grad_accum,
        "effective_batch_size": hparams["dataloader_opts"][
                                    "batch_size"] * args.grad_accum,
        "lora_r": args.lora_r if args.peft == "lora" else None,
        "lora_alpha": args.lora_alpha if args.peft == "lora" else None,
        "lora_dropout": args.lora_dropout if args.peft == "lora" else None,
        "quant_weights_dtype": None,
        "quant_compute_dtype": None,
        "inference_time_sec": None,
        "peak_memory_mb": peak_memory,
        "save_full_model": args.save_full_model,
        "language": args.language,
        "task": args.task,
        "learning_rate": args.lr,
        "seed": args.seed,
        "eval_strategy": args.eval_strategy,
        "eval_steps": args.eval_steps,
        "notes": args.notes,
        "git_commit_hash": get_git_commit_hash(),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    if args.metrics_csv:
        csv_path = Path(args.metrics_csv)
        is_new_file = not csv_path.exists()
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(csv_path, mode='a', header=is_new_file, index=False)

    logger.info(f"Training complete. Artifacts saved to {output_dir}")
    log_gpu_memory("Final GPU Memory")


if __name__ == "__main__":
    main()
