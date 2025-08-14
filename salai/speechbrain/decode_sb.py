import argparse
import os
import json
import time
from typing import Optional

import torch
import torch.nn as nn

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

# Robust imports: work with both `python -m ...` and direct script runs
try:
    from .sb_utils import build_model_from_hparams
    from ..common.text_norm import normalize_atc_text
except Exception:
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from asr_finetune.speechbrain.sb_utils import build_model_from_hparams
    from asr_finetune.common.text_norm import normalize_atc_text

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hparams_file", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--beam_size", type=int, default=8)
    args = p.parse_args()

    with open(args.hparams_file, "r", encoding="utf-8") as f:
        hparams = load_hyperpyyaml(f)

    # Build model and load best checkpoint
    modules, hparams = build_model_from_hparams(hparams, label_encoder=None)  # decoder builds its own encoder from checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpointer = sb.utils.checkpoints.Checkpointer(args.checkpoint_dir, recoverables=modules)
    checkpointer.recover_if_possible()

    # Load manifest
    items = []
    if args.manifest.endswith(".json") or args.manifest.endswith(".jsonl"):
        with open(args.manifest, "r", encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
    else:
        import pandas as pd
        df = pd.read_csv(args.manifest)
        items = df.to_dict("records")

    # Decode greedily for simplicity here
    from speechbrain.dataio.encoder import CategoricalEncoder
    label_encoder = hparams.get("label_encoder")
    if label_encoder is None:
        # Recreate a simple char encoder if not saved
        enc = CategoricalEncoder()
        symbols = [" "] + [chr(c) for c in range(65, 91)] + [str(d) for d in range(10)] + [".", "/"]
        for s in symbols:
            enc.update_from_iterable([s])
        enc.add_unk(); enc.add_bos_eos()
        label_encoder = enc

    modules = {k: v.to(device) if isinstance(v, nn.Module) else v for k, v in modules.items()}
    for m in modules.values():
        if isinstance(m, nn.Module):
            m.eval()

    hyps = []
    with torch.no_grad():
        for ex in items:
            import torchaudio
            sig, sr = torchaudio.load(ex["wav"])
            if sr != 16000:
                sig = torchaudio.functional.resample(sig, sr, 16000)
            if sig.shape[0] > 1:
                sig = torch.mean(sig, dim=0, keepdim=True)
            feats = hparams["compute_features"](sig.to(device))
            feats = hparams["normalize"](feats, torch.tensor([1.0], device=device))
            enc_out = modules["encoder"](feats)
            logits = modules["ctc_lin"](enc_out)
            p_ctc = hparams["log_softmax"](logits)
            greedy = torch.argmax(p_ctc, dim=-1)
            hyp = label_encoder.decode_ndim(greedy)[0] if greedy.ndim > 1 else label_encoder.decode_ndim(greedy)
            hyps.append({"id": ex["id"], "hyp": hyp})

    out_path = os.path.join(args.checkpoint_dir, "decode.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for h in hyps:
            f.write(json.dumps(h) + "\n")
    print(f"Wrote hypotheses to {out_path}")


if __name__ == "__main__":
    main()