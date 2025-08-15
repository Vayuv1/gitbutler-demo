# salai/asr_finetune/speechbrain/decode_sb.py
"""
This script performs inference with a trained SpeechBrain ASR model.
It takes a manifest file as input and produces a decode.jsonl file with
the model's hypotheses.
"""

import argparse
import json
import torch
import sys
from pathlib import Path
from tqdm import tqdm

import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataloader import SaveableDataLoader
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.encoder import CTCTextEncoder

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from asr_finetune.speechbrain.sb_utils import build_asr_model
from asr_finetune.common.text_norm import normalize_transcript


def dataio_prepare_decode(hparams, manifest_file):
    """Prepares the data for decoding."""

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        return sb.dataio.dataio.read_audio(wav)

    dataset = DynamicItemDataset.from_csv(
        csv_path=manifest_file,
        replacements={"data_root": ""},
    )
    dataset.add_dynamic_item(audio_pipeline)
    dataset.set_output_keys(["id", "sig"])
    return dataset


def main():
    parser = argparse.ArgumentParser(description="SpeechBrain ASR Decoding")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory of the trained model.")
    parser.add_argument("--test_manifest", type=str, required=True,
                        help="Path to the manifest file for decoding.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save the decode.jsonl file. Defaults to model_dir/decode.jsonl.")
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam size for decoding. 1 means greedy decoding.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for inference.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    hparams_file = model_dir / "hyperparams.yaml"
    checkpoint_dir = model_dir / "checkpoints"
    tokenizer_file = model_dir / "tokenizer.json"

    output_file = Path(
        args.output_file) if args.output_file else model_dir / "decode.jsonl"

    # Load hyperparameters and tokenizer
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    tokenizer = CTCTextEncoder.load(tokenizer_file)
    hparams["tokenizer"] = tokenizer

    # Load model
    model = build_asr_model(hparams)
    ctc_lin = hparams["ctc_lin"]
    input_linear = hparams["input_linear"]

    # Load the best checkpoint
    checkpointer = sb.utils.checkpoints.Checkpointer(
        checkpoints_dir=checkpoint_dir,
        recoverables={"model": model, "ctc_lin": ctc_lin,
                      "input_linear": input_linear}
    )
    checkpointer.recover(min_key="WER")

    model.to(args.device)
    ctc_lin.to(args.device)
    input_linear.to(args.device)
    model.eval()
    ctc_lin.eval()
    input_linear.eval()

    # Create dataloader
    decode_dataset = dataio_prepare_decode(hparams, args.test_manifest)
    dataloader = SaveableDataLoader(decode_dataset, batch_size=1, num_workers=4)

    # Decoding
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Decoding"):
            batch = batch.to(args.device)
            wavs, wav_lens = batch.sig

            feats = hparams["compute_features"](wavs)
            feats = input_linear(feats)
            out, _ = model(feats)
            logits = ctc_lin(out)
            p_ctc = torch.nn.functional.log_softmax(logits, dim=-1)

            if args.beam_size > 1:
                # Beam search decoding
                beam_searcher = sb.decoders.ctc_beam_search.CTCBeamSearcher(
                    vocab_list=tokenizer.lab2ind.keys(),
                    beam_size=args.beam_size,
                    blank_id=hparams["blank_index"],
                )
                hyps, _, _, _ = beam_searcher(p_ctc, wav_lens)
                hyp = hyps[0][0]
            else:
                # Greedy decoding
                predicted_tokens = torch.argmax(p_ctc, dim=-1)
                hyp = tokenizer.decode_ndim(predicted_tokens)

            results.append({"id": batch.id[0], "hyp": hyp})

    # Write results to file
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Decoding complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()
