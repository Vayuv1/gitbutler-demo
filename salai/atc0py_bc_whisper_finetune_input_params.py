# atc0py_bc_whisper_finetune_input_params.py
# Fine tune Whisper with LoRA on ATC0. Quantized loading, dataset controls, robust path logic, and CSV logging.

import os
import csv
import datetime
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
)
import evaluate
from peft import LoraConfig, get_peft_model


# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine tune Whisper models with LoRA on ATC0 data"
    )
    parser.add_argument(
        "--model_name",
        default="openai/whisper-medium.en",
        choices=["openai/whisper-medium.en", "openai/whisper-large-v3-turbo"],
        help="Pretrained Whisper model to start from.",
    )
    parser.add_argument(
        "--adapter_dir",
        default=None,
        help="Directory to save the LoRA adapter or full model. If omitted a default under finetuned_models is used.",
    )
    parser.add_argument(
        "--save_full_model",
        action="store_true",
        help="Merge LoRA into the base model and save the full model.",
    )
    parser.add_argument(
        "--dataset_part",
        default="base",
        help="ATC0 dataset configuration to use for example base.",
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=1.0,
        help="Fraction of the training split to use between 0 and 1.",
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=200,
        help="Number of validation samples.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of fine tuning epochs.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code for decoder prompt for example en.",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Generation task for decoder prompt.",
    )
    return parser.parse_args()


# ---------------------------
# Auth helper
# ---------------------------
def get_hf_token() -> str:
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise EnvironmentError("HUGGINGFACE_TOKEN not found in .env")
    return token


# ---------------------------
# Data
# ---------------------------
def load_preprocessed_dataset(
    token: str,
    processor: WhisperProcessor,
    dataset_part: str,
    train_fraction: float,
    validation_samples: int,
) -> DatasetDict:
    raw = load_dataset("HF-SaLAI/salai_atc0", dataset_part, token=token)
    dataset = DatasetDict()
    dataset["train"] = raw["train"]
    dataset["validation"] = raw["validation"]

    dataset["validation"] = dataset["validation"].shuffle(seed=42).select(
        range(min(validation_samples, len(dataset["validation"])))
    )

    def is_valid(example):
        return len(processor.tokenizer._normalize(example["text"]).strip()) > 0

    dataset["train"] = dataset["train"].filter(is_valid)
    dataset["validation"] = dataset["validation"].filter(is_valid)

    if 0 < train_fraction < 1.0:
        n = int(len(dataset["train"]) * train_fraction)
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(n))

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        processor.tokenizer.name_or_path
    )
    tokenizer = WhisperTokenizer.from_pretrained(processor.tokenizer.name_or_path)

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]  # typically float32 array
        batch["labels"] = tokenizer(
            processor.tokenizer._normalize(batch["text"]).strip()
        ).input_ids
        return batch

    dataset = dataset.map(
        prepare,
        remove_columns=dataset["train"].column_names,
        num_proc=1,
    )
    return dataset


# ---------------------------
# Collator
# ---------------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    target_dtype: torch.dtype = torch.float32  # will set to float16 on CUDA

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        # Cast audio features to match model compute dtype to avoid fp32 vs fp16 mismatch
        if self.target_dtype is not None:
            batch["input_features"] = batch["input_features"].to(self.target_dtype)

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# ---------------------------
# TrainingArguments builder that works across minor versions
# ---------------------------
def build_training_args(output_dir: str, epochs: int, device: str) -> Seq2SeqTrainingArguments:
    common = dict(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        num_train_epochs=epochs,
        fp16=(device == "cuda"),
        generation_max_length=128,
        logging_steps=10,
        remove_unused_columns=False,
        label_names=["labels"],
        predict_with_generate=True,
        report_to="none",
        save_strategy="no",
    )
    try:
        return Seq2SeqTrainingArguments(evaluation_strategy="epoch", **common)
    except TypeError:
        return Seq2SeqTrainingArguments(eval_strategy="epoch", **common)


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "None"
    print(f"Using device: {device}")

    # Robust path resolution
    script_file = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_file)
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    base_finetuned_dir = os.path.join(repo_root, "finetuned_models")
    os.makedirs(base_finetuned_dir, exist_ok=True)

    model_tail = args.model_name.split("/")[-1]
    default_output = os.path.join(
        base_finetuned_dir,
        f"whisper_lora_{model_tail}_{args.dataset_part}",
    )

    # Resolve adapter_dir
    if args.adapter_dir:
        if os.path.isabs(args.adapter_dir):
            output_dir = args.adapter_dir
        else:
            output_dir = os.path.join(base_finetuned_dir, args.adapter_dir)
    else:
        output_dir = default_output

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("---- PATH DEBUG ----")
    print(f"CWD:            {os.getcwd()}")
    print(f"__file__:       {script_file}")
    print(f"script_dir:     {script_dir}")
    print(f"repo_root:      {repo_root}")
    print(f"finetuned_dir:  {base_finetuned_dir}")
    print(f"OUTPUT DIR:     {output_dir}")
    print("--------------------")

    hf_token = get_hf_token()
    processor = WhisperProcessor.from_pretrained(args.model_name)

    dataset = load_preprocessed_dataset(
        token=hf_token,
        processor=processor,
        dataset_part=args.dataset_part,
        train_fraction=args.train_fraction,
        validation_samples=args.validation_samples,
    )

    # Quantization: 8-bit weights + fp16 compute
    quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=quant_cfg,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # LoRA
    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # Metric tokenizer and WER
    metric_tokenizer = WhisperTokenizer.from_pretrained(args.model_name)
    wer_metric = evaluate.load("wer")

    def compute_metrics_fn(pred):
        labels = pred.label_ids.copy()
        labels[labels == -100] = metric_tokenizer.pad_token_id
        pred_str = metric_tokenizer.batch_decode(
            pred.predictions, skip_special_tokens=True
        )
        label_str = metric_tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    training_args = build_training_args(output_dir, args.epochs, device)

    # Choose collator dtype to match compute
    target_dtype = torch.float16 if device == "cuda" else torch.float32

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor.feature_extractor,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor, target_dtype=target_dtype
        ),
        compute_metrics=compute_metrics_fn,
    )

    # Set decoder prompt to avoid lang_to_id issues
    trainer.model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language, task=args.task
    )

    model.config.use_cache = False
    train_result = trainer.train()
    metrics = train_result.metrics
    print("Training complete.")

    # Save adapter or full model
    if args.save_full_model:
        print("Merging LoRA adapters into the base model and saving full model…")
        model = model.merge_and_unload()
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
    else:
        print("Saving LoRA adapter without merge…")
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    # Evaluate and log memory and time
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    evaluation = trainer.evaluate()
    inference_time = time.time() - t0
    peak_memory = (
        torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None
    )

    # CSV record
    ts = datetime.datetime.now().isoformat()
    record = {
        "timestamp": ts,
        "model": args.model_name,
        "output_dir": output_dir,
        "dataset_part": args.dataset_part,
        "train_fraction": args.train_fraction,
        "validation_samples": args.validation_samples,
        "epochs": args.epochs,
        "wer": evaluation.get("eval_wer"),
        "train_runtime": metrics.get("train_runtime"),
        "train_loss": metrics.get("train_loss"),
        "eval_loss": evaluation.get("eval_loss"),
        "device": device,
        "gpu": gpu_name,
        "batch_size": training_args.per_device_train_batch_size,
        "lora_r": lora_cfg.r,
        "lora_alpha": lora_cfg.lora_alpha,
        "lora_dropout": lora_cfg.lora_dropout,
        "quant_weights_dtype": "int8",
        "quant_compute_dtype": "float16" if device == "cuda" else "float32",
        "inference_time_sec": inference_time,
        "peak_memory_mb": peak_memory,
        "save_full_model": args.save_full_model,
        "language": args.language,
        "task": args.task,
    }

    # CSV in same folder as this script
    csv_path = os.path.join(script_dir, "finetune_log.csv")
    fieldnames = list(record.keys())
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

    print(f"Evaluation and training summary appended to: {csv_path}")


if __name__ == "__main__":
    main()
