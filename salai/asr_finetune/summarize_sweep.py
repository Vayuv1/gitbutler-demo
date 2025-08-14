# summarize_sweep.py
import json
import csv
from datetime import datetime
from pathlib import Path

MASTER = Path("asr_finetune/sweep_runs/sweep_master.csv")
BEST_JSON = Path("asr_finetune/sweep_runs/best_config.json")

def to_float(x, default=float("inf")):
    try: return float(x)
    except: return default

def main():
    if not MASTER.exists():
        print("No master CSV:", MASTER)
        return

    rows = list(csv.DictReader(open(MASTER)))
    if not rows:
        print("Master CSV is empty.")
        return

    ranked = [r for r in rows if r.get("wer")]
    ranked.sort(key=lambda r: (to_float(r["wer"]),
                               to_float(r.get("inference_time_sec")),
                               to_float(r.get("peak_memory_mb"))))

    print(f"Total rows: {len(rows)}  |  Ranked: {len(ranked)}")
    print("\nTop 10 overall:")
    for r in ranked[:10]:
        print({
            "stage": r.get("run_stage"),
            "run_id": r.get("run_id"),
            "model": r.get("model"),
            "train_fraction": r.get("train_fraction"),
            "epochs": r.get("epochs"),
            "wer": r.get("wer"),
            "inference_time_sec": r.get("inference_time_sec"),
            "peak_memory_mb": r.get("peak_memory_mb"),
            "lr": r.get("learning_rate"),
            "lora_r": r.get("lora_r"),
            "lora_alpha": r.get("lora_alpha"),
            "lora_dropout": r.get("lora_dropout"),
            "stdout_log": r.get("stdout_log"),
        })

    if not ranked:
        return

    best = ranked[0]
    cfg = {
        "model_name": best.get("model"),
        "dataset_part": best.get("dataset_part"),
        "train_fraction": best.get("train_fraction"),
        "validation_samples": best.get("validation_samples"),
        "epochs": best.get("epochs"),
        "language": best.get("language", "en"),
        "task": best.get("task", "transcribe"),
        "learning_rate": best.get("learning_rate", "1e-3"),
        "per_device_train_batch_size": best.get("per_device_train_batch_size", "4"),
        "gradient_accumulation_steps": best.get("grad_accum", "2"),
        "lora_r": best.get("lora_r"),
        "lora_alpha": best.get("lora_alpha"),
        "lora_dropout": best.get("lora_dropout"),
        "seed": best.get("seed", "42"),
    }
    BEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_JSON, "w") as f:
        json.dump(cfg, f, indent=2)
    print("\nBest config written to:", BEST_JSON.resolve())

    # Give a unique adapter_dir for the final save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tail = (cfg["model_name"] or "model").split("/")[-1]
    adapter_dir = f"best_{model_tail}_{cfg['dataset_part']}_{ts}"

    cmd = [
        "python asr_finetune/atc0py_bc_whisper_finetune_input_params_sweep.py",
        f'--model_name "{cfg["model_name"]}"',
        f'--dataset_part {cfg["dataset_part"]}',
        f'--train_fraction {cfg["train_fraction"]}',
        f'--validation_samples {cfg["validation_samples"]}',
        f'--epochs {cfg["epochs"]}',
        f'--language {cfg["language"]}',
        f'--task {cfg["task"]}',
        f'--learning_rate {cfg["learning_rate"]}',
        f'--per_device_train_batch_size {cfg["per_device_train_batch_size"]}',
        f'--gradient_accumulation_steps {cfg["gradient_accumulation_steps"]}',
        f'--lora_r {cfg["lora_r"]}',
        f'--lora_alpha {cfg["lora_alpha"]}',
        f'--lora_dropout {cfg["lora_dropout"]}',
        f'--seed {cfg["seed"]}',
        f'--adapter_dir {adapter_dir}',
        "--save_full_model",
        # NOTE: no --skip_save here, we WANT to save the best
    ]
    print("\nRe-run best with full save:\n" + " \\\n  ".join(cmd))
    print("\nAdapter will be saved under finetuned_models/", adapter_dir)

if __name__ == "__main__":
    main()
