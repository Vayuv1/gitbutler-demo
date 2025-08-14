# asr_finetune/finetune_whisper_atc0_base.py

"""Whisper fine-tuning utility.

The script fine‑tunes an arbitrary Whisper model on the ATC0 dataset using
LoRA adapters. It can operate on the default `openai/whisper-medium.en`
model or on a larger variant such as `openai/whisper-large-v3-turbo`. Models
are loaded with 8‑bit weights and 16‑bit compute to reduce memory usage. The
training/evaluation summary is appended to a CSV log that now also captures
inference time, memory consumption and various runtime options.
"""

import argparse
import csv
import datetime
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import psutil
import torch
from dotenv import load_dotenv
from datasets import DatasetDict, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

import evaluate
from peft import LoraConfig, get_peft_model


# Load Hugging Face Token
def get_hf_token():
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise EnvironmentError("HUGGINGFACE_TOKEN not found in .env")
    return token


# Load and Preprocess Dataset
def load_preprocessed_dataset(
    token: str,
    processor: WhisperProcessor,
    model_name: str,
    dataset_part: str,
    dataset_fraction: float,
) -> DatasetDict:
    """Load ATC0 dataset, filter/prepare it and optionally downsample.

    Parameters
    ----------
    token:
        Hugging Face access token.
    processor:
        ``WhisperProcessor`` used for text normalisation.
    model_name:
        Identifier of the Whisper model providing tokenizer & feature extractor.
    dataset_part:
        Which part of the dataset to use (e.g. ``base`` or ``all``).
    dataset_fraction:
        Fraction of each split to retain.
    """

    raw = load_dataset("HF-SaLAI/salai_atc0", dataset_part, token=token)
    dataset = DatasetDict()
    dataset["train"] = raw["train"]
    dataset["validation"] = raw["validation"].shuffle(seed=42)

    if dataset_fraction < 1.0:
        train_size = max(1, int(len(dataset["train"]) * dataset_fraction))
        val_size = max(1, int(len(dataset["validation"]) * dataset_fraction))
        dataset["train"] = dataset["train"].select(range(train_size))
        dataset["validation"] = dataset["validation"].select(range(val_size))
    else:
        dataset["validation"] = dataset["validation"].select(
            range(min(200, len(dataset["validation"])))
        )

    def is_valid(example):
        return len(processor.tokenizer._normalize(example["text"]).strip()) > 0

    dataset["train"] = dataset["train"].filter(is_valid)
    dataset["validation"] = dataset["validation"].filter(is_valid)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name)

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = tokenizer(
            processor.tokenizer._normalize(batch["text"]).strip()
        ).input_ids
        return batch

    dataset = dataset.map(
        prepare, remove_columns=dataset["train"].column_names, num_proc=1
    )
    return dataset


# Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# Evaluation Metrics
def compute_metrics_factory(tokenizer: WhisperTokenizer):
    def compute_metrics(pred):
        label_ids = pred.label_ids
        pred_ids = pred.predictions

        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer_metric = evaluate.load("wer")
        return {
            "wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        }

    return compute_metrics


# Main Training Function
def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on ATC0")
    default_output = os.path.join(
        os.path.dirname(__file__), "..", "finetuned_models", "whisper_lora_atc0_base"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/whisper-medium.en",
        help="Whisper model identifier",
    )
    parser.add_argument(
        "--adapter-save-path",
        type=str,
        default=None,
        help="Directory to save the trained LoRA adapter",
    )
    parser.add_argument(
        "--save-full-model",
        action="store_true",
        help="Merge LoRA and save the full model",
    )
    parser.add_argument(
        "--dataset-part",
        type=str,
        default="base",
        help="Dataset configuration to use (e.g. base, all)",
    )
    parser.add_argument(
        "--dataset-fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset to use (0-1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output,
        help="Where to save the full model if requested",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "None"
    print(f"Using device: {device}")

    hf_token = get_hf_token()
    processor = WhisperProcessor.from_pretrained(args.model_name)
    dataset = load_preprocessed_dataset(
        hf_token, processor, args.model_name, args.dataset_part, args.dataset_fraction
    )

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    base_model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        load_in_8bit=device == "cuda",
        device_map="auto" if device == "cuda" else None,
    ).to(device)
    base_model.config.forced_decoder_ids = None
    base_model.config.suppress_tokens = []

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    os.makedirs(args.output_dir, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        num_train_epochs=5,
        fp16=device == "cuda",
        eval_strategy="epoch",
        generation_max_length=128,
        logging_steps=10,
        remove_unused_columns=False,
        label_names=["labels"],
        predict_with_generate=True,
        report_to="none",
        save_strategy="no",
    )

    tokenizer = WhisperTokenizer.from_pretrained(args.model_name)
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor.feature_extractor,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics_factory(tokenizer),
    )

    model.config.use_cache = False
    train_result = trainer.train()
    metrics = train_result.metrics

    if args.adapter_save_path:
        os.makedirs(args.adapter_save_path, exist_ok=True)
        model.save_pretrained(args.adapter_save_path)

    if args.save_full_model:
        print("Training complete. Saving final model locally...")
        merged = model.merge_and_unload()
        merged.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}")

    start_eval = time.time()
    evaluation = trainer.evaluate()
    inference_time = time.time() - start_eval

    max_memory = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if device == "cuda"
        else psutil.Process().memory_info().rss / (1024 ** 2)
    )

    timestamp = datetime.datetime.now().isoformat()
    evaluation_record = {
        "timestamp": timestamp,
        "model": args.model_name,
        "output_dir": args.output_dir,
        "epochs": training_args.num_train_epochs,
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
        "adapter_path": args.adapter_save_path,
        "save_full_model": args.save_full_model,
        "dataset_part": args.dataset_part,
        "dataset_fraction": args.dataset_fraction,
        "inference_time": inference_time,
        "max_memory_mb": max_memory,
    }

    csv_log_path = os.path.join(args.output_dir, "finetune_log.csv")
    fieldnames = list(evaluation_record.keys())
    file_exists = os.path.isfile(csv_log_path)

    with open(csv_log_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(evaluation_record)

    print("Evaluation and training summary appended to:", csv_log_path)


if __name__ == "__main__":
    main()
