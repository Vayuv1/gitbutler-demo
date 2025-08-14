import os
import csv
import argparse
import hashlib
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
import numpy as np
import soundfile as sf
import torchaudio
import torch

try:
    from ..common.text_norm import normalize_atc_text
except Exception:
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from asr_finetune.common.text_norm import normalize_atc_text



def write_split(ds, out_audio_dir: Path, manifest_path: Path, split: str, resample_hz: int = 16000):
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "wav", "duration", "transcript"])
        w.writeheader()
        for i, ex in enumerate(ds):
            aud = ex["audio"]  # {'array': np.ndarray, 'sampling_rate': int}
            arr = aud["array"].astype(np.float32)
            sr = int(aud["sampling_rate"]) if isinstance(aud["sampling_rate"], (int, np.integer)) else 16000
            # resample to 16 kHz mono
            wav = torch.from_numpy(arr).unsqueeze(0)
            if wav.dim() == 2 and wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)  # mono
            if sr != resample_hz:
                wav = torchaudio.functional.resample(wav, sr, resample_hz)
                sr = resample_hz
            wav_np = wav.squeeze(0).cpu().numpy().astype(np.float32)
            text = normalize_atc_text(ex.get("text", ""))
            # make a stable id
            raw_id = ex.get("id") or f"{split}-{i}"
            h = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
            wav_path = out_audio_dir / f"{split}_{h}.wav"
            sf.write(wav_path, wav_np, sr)
            duration = len(wav_np) / sr
            w.writerow({
                "id": f"{split}_{h}",
                "wav": str(wav_path.resolve()),
                "duration": round(duration, 3),
                "transcript": text,
            })


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=["base", "part2", "part3"], default="base")
    p.add_argument("--out_dir", default="data/atc0")
    p.add_argument("--train_fraction", type=float, default=1.0)
    p.add_argument("--validation_samples", type=int, default=0)
    args = p.parse_args()

    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if token is None:
        raise RuntimeError("Set HUGGINGFACE_TOKEN in your environment or .env file.")

    ds = load_dataset("HF-SaLAI/salai_atc0", args.config, use_auth_token=True)

    base_out = Path(args.out_dir) / args.config
    base_out.mkdir(parents=True, exist_ok=True)

    # Train split
    train_ds = ds["train"]
    if 0 < args.train_fraction < 1.0:
        n = int(len(train_ds) * args.train_fraction)
        train_ds = train_ds.select(range(n))
    write_split(train_ds, base_out / "train_wav", base_out / "train.csv", split="train")

    # Validation split
    if "validation" in ds:
        valid_ds = ds["validation"]
        if args.validation_samples and args.validation_samples > 0:
            valid_ds = valid_ds.select(range(min(args.validation_samples, len(valid_ds))))
        write_split(valid_ds, base_out / "valid_wav", base_out / "valid.csv", split="valid")

    # Test split
    if "test" in ds:
        test_ds = ds["test"]
        write_split(test_ds, base_out / "test_wav", base_out / "test.csv", split="test")

    print(f"Wrote manifests under {base_out}")


if __name__ == "__main__":
    main()