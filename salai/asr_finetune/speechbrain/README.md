# SpeechBrain Finetuning on ATC0 (Conformer‑CTC)

This package provides a **SpeechBrain** implementation for fine‑tuning a Conformer‑CTC ASR model on the **ATC0** dataset. It mirrors the UX of your Whisper scripts while following SpeechBrain best practices and keeping the outputs compatible with your sweep/summary tooling.

---
## 0) Requirements
- Python 3.10+
- GPU with CUDA (recommended) and at least ~12 GB VRAM for the baseline
- PyTorch + torchaudio, SpeechBrain, HyperPyYAML, datasets, python‑dotenv, soundfile, sentencepiece (optional)

Install (example):
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install speechbrain hyperpyyaml datasets python-dotenv soundfile sentencepiece
```

---
## 1) File overview
- **`train_sb.py`** — main trainer. Builds tokenizer & datasets from manifests, constructs Conformer from `hparams_sb.yaml`, optional LoRA, trains with AMP, evaluates WER, logs metrics, saves checkpoints.
- **`decode_sb.py`** — offline decoder. Loads a checkpoint and produces `decode.jsonl` hypotheses for a given manifest.
- **`hparams_sb.yaml`** — model/feature hyper‑parameters (Fbank, SpecAugment, Conformer shape, CTC loss/beam).
- **`sb_utils.py`** — helpers: LoRA wrapper for Linear layers, model builder, model card writer.
- **`../common/text_norm.py`** — ATC‑friendly normalization (UPPERCASE, digits kept, only `.` and `/`).
- **`../common/metrics.py`** — WER computation (greedy/beam) using SpeechBrain’s metrics.
- **`../data_prep/atc0_hf_to_manifests.py`** — converts **HF‑SaLAI/salai_atc0** to local WAVs + CSV manifests.

Folder convention for outputs:
```
salai/finetuned_models/speechbrain/<model_name>_<YYYYmmdd_HHMMSS>/
  checkpoints/   train_log.txt   metrics.json   model_card.json
```

---
## 2) Data preparation (one‑time)
**Why manifests?** Stable file paths, one‑time normalization, 16 kHz mono, engine‑agnostic I/O, faster epochs, and reproducibility.

**Create manifests/WAVs**
```bash
export HUGGINGFACE_TOKEN=***your_token***
python asr_finetune/data_prep/atc0_hf_to_manifests.py --config base --out_dir data/atc0
# quick sanity subset
python asr_finetune/data_prep/atc0_hf_to_manifests.py --config base --out_dir data/atc0 \
  --train_fraction 0.1 --validation_samples 200
```
Outputs:
```
data/atc0/base/train.csv   valid.csv   test.csv
```
Each CSV row has: `id, wav, duration, transcript`.

**Script flags** (`atc0_hf_to_manifests.py`):
- `--config` (base | part2 | part3) — which ATC0 configuration to export.
- `--out_dir` — root folder to write audio + manifests (default `data/atc0`).
- `--train_fraction` — use first fraction of train split (0<val≤1.0). Useful for small test runs.
- `--validation_samples` — cap number of validation items (0 means all).

> Tip: To *merge* `part2` / `part3` into training, run the script for each config, then concatenate their `train.csv` files into a single manifest.

---
## 3) Training — `train_sb.py`
Runs a baseline Conformer‑CTC with AMP and optional LoRA. Compatible with your sweep CSV when `--metrics_csv` is provided.

### Minimal example
```bash
python asr_finetune/speechbrain/train_sb.py \
  --train_manifest data/atc0/base/train.csv \
  --valid_manifest data/atc0/base/valid.csv \
  --test_manifest  data/atc0/base/test.csv \
  --output_dir salai/finetuned_models/speechbrain \
  --dataset_part base --epochs 5 --lr 1e-3 \
  --batch_size 8 --grad_accum 1 --beam_size 8 \
  --notes "SB baseline ATC0 base"
```

### Flags (by group)
**Data**
- `--train_manifest` — path to train CSV/JSONL.
- `--valid_manifest` — path to validation CSV/JSONL.
- `--test_manifest` — path to test CSV/JSONL (optional).
- `--output_dir` — base directory for run folders and checkpoints.
- `--dataset_part` — short label stored in metrics (e.g., `base`, `part2`).
- `--train_fraction` — recorded in metrics for parity with Whisper sweeps (no slicing occurs here; slicing is done by data prep).
- `--validation_samples` — recorded in metrics (no slicing occurs here; slicing is done by data prep).
- `--train_subset` — **runtime** sampler for quick iterations (uses only the first N rows of the train manifest).
- `--skip_test` — compute only validation WER; skip test.

**Optimization**
- `--epochs` — number of epochs.
- `--lr` — learning rate (AdamW).
- `--batch_size` — target batch size used by a **dynamic length‑based** sampler.
- `--grad_accum` — accumulation steps to reach a desired effective batch size.
- `--seed` — random seed.

**Model**
- `--model_name` — free‑form tag written into the run folder/CSV (e.g., `sb-conformer-ctc`).
- `--hparams_file` — path to `hparams_sb.yaml`.

**Tokenizer**
- `--tokenizer_type` — `char` (default) or `spm`.
- `--spm_model` — path to a SentencePiece model when `--tokenizer_type spm`.

**PEFT (LoRA)**
- `--peft` — `none` (default) or `lora`.
- `--lora_r` — LoRA rank.
- `--lora_alpha` — LoRA scaling.
- `--lora_dropout` — LoRA dropout.

**Decoding**
- `--beam_size` — beam width for CTC decoding in evaluation (use 1 for greedy).
- `--lm_weight` — reserved; external LM not used in the baseline.

**Logging / meta**
- `--run_id` — custom identifier (default is timestamp `YYYYmmdd_HHMMSS`).
- `--run_stage` — e.g., `single`, `coarse`, `refine` (recorded in CSV).
- `--metrics_csv` — if provided, append one row using the **same header** as your Whisper sweeps (see §5).
- `--notes` — free text saved in `metrics.json` (not written to CSV).
- `--save_full_model` — boolean flag recorded in CSV (baseline saves checkpoints regardless).
- `--language`, `--task` — recorded for parity with Whisper.
- `--eval_strategy`, `--eval_steps` — recorded for parity with Whisper sweeps.

### Outputs per run
- `train_log.txt` — SpeechBrain training log
- `checkpoints/` — SB Checkpointer artifacts
- `metrics.json` — one JSON with WER, timings, memory, and flags for this run
- `model_card.json` — minimal card with engine/model/tokenizer/PEFT info

---
## 4) Decoding — `decode_sb.py`
Produces per‑utterance hypotheses for a manifest using a saved checkpoint.

**Example**
```bash
python asr_finetune/speechbrain/decode_sb.py \
  --hparams_file asr_finetune/speechbrain/hparams_sb.yaml \
  --checkpoint_dir salai/finetuned_models/speechbrain/sb-conformer-ctc_YYYYmmdd_HHMMSS \
  --manifest data/atc0/base/test.csv \
  --beam_size 8
```
**Flags**
- `--hparams_file` — YAML to reconstruct features/model.
- `--checkpoint_dir` — run directory containing SB checkpoints.
- `--manifest` — CSV/JSONL with `id,wav,duration,transcript`.
- `--beam_size` — beam width (1 = greedy).

**Output**
- `decode.jsonl` in `checkpoint_dir`, lines like: `{ "id": "...", "hyp": "..." }`.

---
## 5) CSV compatibility (sweep/summary)
When `--metrics_csv` is provided to **`train_sb.py`**, the trainer appends **exactly** the Whisper sweep header:
```
timestamp, run_id, run_stage, model, output_dir, dataset_part, train_fraction, validation_samples,
epochs, wer, train_runtime, train_loss, eval_loss, device, gpu, per_device_train_batch_size,
grad_accum, effective_batch_size, lora_r, lora_alpha, lora_dropout, quant_weights_dtype,
quant_compute_dtype, inference_time_sec, peak_memory_mb, save_full_model, language, task,
learning_rate, seed, eval_strategy, eval_steps
```
Fill‑ins for SpeechBrain baseline:
- `quant_weights_dtype` = `fp32`
- `quant_compute_dtype` = `float16` on GPU (AMP) or `float32` on CPU
- `wer` = validation WER
- `inference_time_sec` = time to compute validation WER after training
- `peak_memory_mb` = CUDA peak allocated memory

> For single runs where you do **not** want to write the master CSV, simply omit `--metrics_csv`. You will still get `metrics.json` in the run folder.

---
## 6) Hyper‑parameters — `hparams_sb.yaml`
Main blocks:
- **Features** — 80‑mel Fbank at 16 kHz, 25 ms window / 10 ms hop.
- **Normalization** — global CMVN.
- **SpecAugment** — frequency/time masking (disable or tune as needed).
- **Model** — Conformer encoder (12 layers, d_model=256, heads=4, dropout=0.1). Adjust for your GPU.
- **CTC head/loss** — Linear to vocab size (auto‑resized from tokenizer), CTC loss with blank index 0.
- **CTCBeamSearcher** — used for validation decoding (beam size set here, overridden by `--beam_size`).

> Later, you can add streaming knobs (chunk size, left/right context) to the YAML and surface them via flags.

---
## 7) LoRA details
- Enabled via `--peft lora` with `--lora_r`, `--lora_alpha`, `--lora_dropout`.
- Injected broadly into **Linear layers inside the Conformer encoder** to keep it simple and robust. You can narrow targets after profiling.
- Baseline recommendation: first run **full fine‑tune** (no PEFT) to validate pipeline, then enable LoRA.

---
## 8) Examples
**Baseline with CSV logging**
```bash
python asr_finetune/speechbrain/train_sb.py \
  --train_manifest data/atc0/base/train.csv \
  --valid_manifest data/atc0/base/valid.csv \
  --test_manifest  data/atc0/base/test.csv \
  --output_dir salai/finetuned_models/speechbrain \
  --dataset_part base --epochs 5 --lr 1e-3 \
  --batch_size 8 --grad_accum 4 --beam_size 8 \
  --metrics_csv asr_finetune/sweep_runs/sweep_master.csv \
  --notes "SB baseline ATC0 base"
```
**LoRA run**
```bash
python asr_finetune/speechbrain/train_sb.py ... \
  --peft lora --lora_r 32 --lora_alpha 64 --lora_dropout 0.05
```
**Quick sanity** (no CSV, small subset)
```bash
python asr_finetune/speechbrain/train_sb.py ... \
  --train_subset 500 --skip_test --notes "smoke test"
```
**SentencePiece tokenizer**
```bash
python asr_finetune/speechbrain/train_sb.py ... \
  --tokenizer_type spm --spm_model path/to/model.spm
```

---
## 9) Troubleshooting
- **HF token** — ensure `HUGGINGFACE_TOKEN` (or `HF_TOKEN`) is set before data prep.
- **CUDA OOM** — lower `--batch_size`, raise `--grad_accum`, or reduce model size in YAML.
- **WER oddities** — confirm the same normalization across engines. This repo’s `text_norm.py` uppercases, keeps digits, and only permits `.` and `/`.
- **Slow epochs** — manifests avoid on‑the‑fly decoding; still, you can use `--train_subset` to iterate faster.

---
## 10) Repro tips
- Record the exact git commit and `metrics.json` alongside the run folder.
- Fix seeds (`--seed`) and avoid changing `text_norm.py` mid‑project to keep WER comparable.
- Keep a global `sweep_master.csv` and let single runs opt‑in via `--metrics_csv`.
