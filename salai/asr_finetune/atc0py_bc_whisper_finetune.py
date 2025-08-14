# asr_finetune/finetune_whisper_atc0_base.py

"""
Fine-tunes the openai/whisper-medium.en model on the ATC0 dataset using LoRA.
Includes preprocessing, evaluation, and local saving. GPU support is enabled.
"""

import os
import csv
import json
import datetime
import platform
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
    Seq2SeqTrainer
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
def load_preprocessed_dataset(token: str, processor) -> DatasetDict:
    raw = load_dataset("HF-SaLAI/salai_atc0", "base", token=token)
    dataset = DatasetDict()
    dataset["train"] = raw["train"]
    dataset["validation"] = raw["validation"].shuffle(seed=42).select(range(200))

    def is_valid(example):
        return len(processor.tokenizer._normalize(example["text"]).strip()) > 0

    dataset["train"] = dataset["train"].filter(is_valid)
    dataset["validation"] = dataset["validation"].filter(is_valid)

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium.en")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium.en")

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(
            processor.tokenizer._normalize(batch["text"]).strip()).input_ids
        return batch

    dataset = dataset.map(prepare, remove_columns=dataset["train"].column_names, num_proc=1)
    return dataset

# Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]])\
            -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# Evaluation Metrics
def compute_metrics(pred):
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium.en")
    label_ids = pred.label_ids
    pred_ids = pred.predictions

    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer_metric = evaluate.load("wer")
    return {"wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str)}

# Main Training Function
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "None"
    print(f"Using device: {device}")

    hf_token = get_hf_token()
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
    dataset = load_preprocessed_dataset(hf_token, processor)

    base_model = (WhisperForConditionalGeneration
                  .from_pretrained("openai/whisper-medium.en").to(device))
    base_model.config.forced_decoder_ids = None
    base_model.config.suppress_tokens = []

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    output_dir = os.path.join(os.path.dirname(__file__), "..",
                              "finetuned_models", "whisper_lora_atc0_base")
    os.makedirs(output_dir, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        num_train_epochs=5,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        generation_max_length=128,
        logging_steps=10,
        remove_unused_columns=False,
        label_names=["labels"],
        predict_with_generate=True,
        report_to="none",
        save_strategy="no"
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor.feature_extractor,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics
    )

    model.config.use_cache = False
    train_result = trainer.train()
    metrics = train_result.metrics

    print("Training complete. Saving final model locally...")
    model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    evaluation = trainer.evaluate()
    timestamp = datetime.datetime.now().isoformat()
    evaluation_record = {
        "timestamp": timestamp,
        "model": "openai/whisper-medium.en",
        "output_dir": output_dir,
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
        "lora_dropout": lora_cfg.lora_dropout
    }

    csv_log_path = os.path.join(output_dir, "finetune_log.csv")
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
