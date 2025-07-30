# Finetuning Whisper with ATC0

The code embedded in this markdown file is supposed to run as cells in a Jupyter notebook. This can be done using Jupyter Lab.

## Finetuning the medium-en model with the base configuration of the ATC0 dataset

### Using the HF token

```python
import os
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
```

### Defining the base model name or path

```python
model_name_or_path = "openai/whisper-medium.en"
```

### Loading the dataset

```python
from datasets import DatasetDict, load_dataset

atc0 = load_dataset("HF-SaLAI/salai_atc0", "base", token=hf_token)

dataset = DatasetDict()
dataset["train"] = atc0["train"]
shuffled_dataset = atc0["validation"].shuffle(seed=42)
dataset["validation"] = shuffled_dataset.select(range(200))

print(dataset)
```

### Creating a processor

```python
import torch

print(f"{torch.cuda.is_available() = }")
```

```python
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(model_name_or_path)
```

### Filtering the test dataset

Some examples will have an empty string after normalization, which will cause issues with the WER calculation. Here, we remove these examples.

```python
def is_transcript_empty(transcript):
    normalized_transcript = processor.tokenizer._normalize(transcript)
    return len(normalized_transcript) > 0

dataset["train"] = dataset["train"].filter(is_transcript_empty,
        input_columns=["text"])
dataset["validation"] = dataset["validation"].filter(is_transcript_empty,
        input_columns=["text"])
print(dataset)
```


### Preparing Feature Extractor and Tokenizer

```python
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path)
```

### Creating input features from audio data

```python
def prepare_dataset(batch):
    # compute log-Mel input features from input audio array
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(
        processor.tokenizer._normalize(batch["text"])).input_ids
    return batch

dataset = dataset.map(
    prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=1)

print(dataset)
```

### Training and Evaluation

#### Define a Data Collator

```python
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
```

```python
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
```

#### Define Evaluation Metrics

```python
import evaluate

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
```

#### Load a pre-trained checkpoint

```python
from transformers import WhisperForConditionalGeneration

base_model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path).to("cuda")
```

Override generation arguments - no tokens are forced as decoder outputs (see [`forced_decoder_ids`](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.forced_decoder_ids)), no tokens are suppressed during generation (see [`suppress_tokens`](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.suppress_tokens)):

```python
base_model.config.forced_decoder_ids = None
base_model.config.suppress_tokens = []
```


#### Apply LoRA


```python
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

model = get_peft_model(base_model, config)

model.print_trainable_parameters()
```

#### Define the Training Configuration

```python
from transformers import Seq2SeqTrainingArguments

output_dir = 'whisper-lora-atc0-base2'

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,          # change to a repo name of your choice
    per_device_train_batch_size=4, # increase to 16 for larger datasets
    gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-3,
    report_to="none",
    # warmup_steps=50,
    num_train_epochs=5,
    eval_strategy="epoch",
    fp16=True,
    per_device_eval_batch_size=1,
    generation_max_length=128,
    logging_steps=1,
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
    predict_with_generate=True,
    save_steps=0.2, #if you wish to save checkpoints
)
```

```python
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
```

#### Train the adapter

```python
trainer.train()
```

Epoch 	Training Loss 	Validation Loss 	Wer
1 	0.537300 	0.593508 	19.382504
2 	0.139400 	0.497896 	17.724414
3 	0.180000 	0.436374 	15.666095
4 	0.007200 	0.431122 	14.351058
5 	0.010100 	0.415403 	12.349914


```python
adapter_to_push = f"{output_dir}/checkpoint-4255"
print(f"{adapter_to_push = }")
```

```python
from transformers import WhisperForConditionalGeneration
from peft import PeftModel

# base_model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, load_in_8bit=False, device_map="auto")
base_model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path).to("cuda")
model = PeftModel.from_pretrained(
    base_model,
    adapter_to_push,
)
print(f"{model = } \n\n")

model = model.merge_and_unload()
print(f"{model = } \n\n")

# model.merge_and_unload() merges the adapter parameters with the base model parameters and unloads the adapter. This typically results in a standard model that can be used without needing the PEFT infrastructure.
```

model1 = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, load_in_8bit=False, device_map="auto")
model2 = WhisperForConditionalGeneration.from_pretrained(model_name_or_path).to("cuda")

Note of the code:

-   They both use the default 16-bit coeficient for the checkpoint.
-   The first one uses `device_map="auto"` to specify to automatically distributes model layers across available hardware, which can optimize performance and memory usage, especially in multi-GPU setups.
-   The second uses `.to("cuda")` to specify to move the entire model to a single GPU, which is straightforward but may not utilize multiple GPUs or balance resources as effectively.
-   The second approach is the preferred way for the single GPU case. If we use multiple GPUs, we can use the first one.

```python
trained_model_name = f"{output_dir}/whisper-medium.en-finetuned-on-atc0-base2"
# whisper-lora-atc0-base/whisper-medium.en-finetuned-on-atc0-base
model.save_pretrained(trained_model_name)
processor.save_pretrained(trained_model_name)
```

```python
print(trained_model_name)
```
