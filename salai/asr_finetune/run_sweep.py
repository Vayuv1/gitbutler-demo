# run_sweep.py
import os
import csv
import time
import json
import subprocess
from pathlib import Path

# Absolute paths
THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parent.parent
SCRIPT_PATH = PROJECT_ROOT / "asr_finetune" / "atc0py_bc_whisper_finetune_input_params_sweep.py"

if not SCRIPT_PATH.exists():
    raise FileNotFoundError(f"Training script not found: {SCRIPT_PATH}")

RUN_ROOT = PROJECT_ROOT / "asr_finetune" / "sweep_runs"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
TMP_CSV_DIR = RUN_ROOT / "tmp_metrics"
TMP_CSV_DIR.mkdir(exist_ok=True)
MASTER_CSV = RUN_ROOT / "sweep_master.csv"
PER_RUN_STDOUT = RUN_ROOT / "stdout_logs"
PER_RUN_STDOUT.mkdir(exist_ok=True)

DATASET_PART = "base"
VAL_SAMPLES = 200
LANGUAGE = "en"
TASK = "transcribe"

# Heuristic for 4090: target eff batch ~32–36 with int8 weights + fp16 compute
def model_defaults(model_name: str):
    if "large-v3-turbo" in model_name:
        return dict(bs=6, grad_accum=6, lrs=[5e-4, 1e-3])    # eff ≈ 36
    else:
        return dict(bs=8, grad_accum=4, lrs=[7.5e-4, 1e-3, 1.5e-3])  # eff ≈ 32

def tail_log(path: Path, n=80):
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        print("---- last log lines ----")
        print("".join(lines[-n:]))
        print("---- end log ----")
    except Exception as e:
        print(f"(could not read log {path}: {e})")

def append_master(row: dict):
    write_header = not MASTER_CSV.exists()
    with open(MASTER_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def read_single_row_csv(csv_path: Path):
    if not csv_path.exists():
        return None
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None

def build_cmd(p, run_id, run_stage, metrics_csv):
    cmd = [
        "python", str(SCRIPT_PATH),
        "--model_name", p["model_name"],
        "--dataset_part", DATASET_PART,
        "--train_fraction", str(p["train_fraction"]),
        "--validation_samples", str(VAL_SAMPLES),
        "--epochs", str(p["epochs"]),
        "--language", LANGUAGE,
        "--task", TASK,
        "--per_device_train_batch_size", str(p["bs"]),
        "--gradient_accumulation_steps", str(p["grad_accum"]),
        "--learning_rate", str(p["lr"]),
        "--lora_r", str(p["lora_r"]),
        "--lora_alpha", str(p["lora_alpha"]),
        "--lora_dropout", str(p["lora_dropout"]),
        "--seed", "42",
        "--num_proc", "1",
        "--eval_strategy", "epoch",
        "--eval_steps", "100",
        "--run_stage", run_stage,
        "--run_id", run_id,
        "--skip_save",
        "--metrics_csv", str(metrics_csv),
    ]
    return cmd

def run_one(run_id: str, params: dict, run_stage: str):
    stdout_file = PER_RUN_STDOUT / f"{run_id}.log"
    metrics_csv = TMP_CSV_DIR / f"{run_id}.csv"
    cmd = build_cmd(params, run_id, run_stage, metrics_csv)
    print(f"[{run_stage}][{run_id}] {json.dumps(params)}")

    t0 = time.time()
    with open(stdout_file, "w") as out:
        out.write("CMD: " + " ".join(cmd) + "\n")
        out.flush()
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT))
        ret = proc.wait()
    dur = time.time() - t0
    print(f"[{run_stage}][{run_id}] finished code={ret} in {dur/60:.1f} min")

    if ret != 0:
        tail_log(stdout_file)
        return

    row = read_single_row_csv(metrics_csv)
    if row:
        row = dict(row)
        row["stdout_log"] = str(stdout_file)
        append_master(row)
        # optional: cleanup per-run metrics
        # metrics_csv.unlink(missing_ok=True)
    else:
        print(f"[{run_stage}][{run_id}] WARNING: no metrics row found at {metrics_csv}")

def coarse_phase():
    runs = []
    models = ["openai/whisper-medium.en", "openai/whisper-large-v3-turbo"]

    # Pruned coarse screen: 25–50% train, 1–2 epochs, slim LR + LoRA grid
    for m in models:
        defaults = model_defaults(m)
        for train_fraction in [0.25, 0.5]:
            for epochs in [1, 2]:
                for lr in defaults["lrs"]:
                    for (lora_r, lora_alpha, lora_dropout) in [
                        (16, 32, 0.05),
                        (32, 64, 0.05),
                    ]:
                        runs.append(dict(
                            model_name=m,
                            train_fraction=train_fraction,
                            epochs=epochs,
                            lr=lr,
                            bs=defaults["bs"],
                            grad_accum=defaults["grad_accum"],
                            lora_r=lora_r,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout,
                        ))
    print(f"Coarse runs: {len(runs)}")
    for i, p in enumerate(runs, start=1):
        run_one(f"coarse_{i:03d}", p, "coarse")

def topk_from_master(k_per_model=4):
    if not MASTER_CSV.exists():
        return []
    rows = list(csv.DictReader(open(MASTER_CSV)))
    coarse = [r for r in rows if r.get("run_stage") == "coarse" and r.get("wer")]
    def to_float(x, default=float("inf")):
        try: return float(x)
        except: return default
    by_model = {}
    for r in coarse:
        by_model.setdefault(r["model"], []).append(r)
    selected = []
    for model, entries in by_model.items():
        entries.sort(key=lambda r: (to_float(r.get("wer")), to_float(r.get("inference_time_sec"))))
        selected.extend(entries[:k_per_model])
    return selected

def refine_phase():
    selected = topk_from_master(k_per_model=4)
    print(f"Refine seeds from coarse: {len(selected)}")
    idx = 1
    for base in selected:
        m = base["model"]
        defaults = model_defaults(m)
        params = dict(
            model_name=m,
            train_fraction=1.0,
            epochs=5,
            lr=float(base.get("learning_rate", defaults["lrs"][0])),
            bs=defaults["bs"],
            grad_accum=defaults["grad_accum"],
            lora_r=int(float(base.get("lora_r", 32))),
            lora_alpha=int(float(base.get("lora_alpha", 64))),
            lora_dropout=float(base.get("lora_dropout", 0.05)),
        )
        run_one(f"refine_{idx:03d}", params, "refine")
        idx += 1

def main():
    coarse_phase()
    refine_phase()
    print("Sweep complete.")
    print("Master CSV:", MASTER_CSV.resolve())
    print("Stdout logs:", PER_RUN_STDOUT.resolve())
    print("Temp metrics:", TMP_CSV_DIR.resolve())

if __name__ == "__main__":
    main()
