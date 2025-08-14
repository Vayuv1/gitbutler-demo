<!-- bgn hidden -->

```toc
style: number
min_depth: 1
max_depth: 6
```

<!-- end hidden -->

# Finetuning Whisper with the ATC0 Dataset

The code embedded in this markdown file is supposed to run as cells in a Jupyter notebook. This can be done using Jupyter Lab.

## Finetuning the medium-en model with the entire ATC0 dataset

### Using the HF token

```python
import os
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
# print(hf_token)
```

### Defining all parameters

```python
# the base model name or path
model_name_or_path = "openai/whisper-medium.en"
output_dir = "whisper-lora-atc0-all"
org = "HF-SaLAI"
# trained_model_name = "whisper-medium.en-finetuned-on-atc0-all"
trained_model_name = "whisper-medium-atc0.en"

adapter_to_choose = f"{output_dir}/checkpoint-28120"
trained_model_local = output_dir + '/' + trained_model_name
trained_model_repo = org + '/' + trained_model_name
```

Epoch 	Training Loss 	Validation Loss 	Wer
1 	0.141900 	0.680747 	28.078818
2 	0.104900 	0.630345 	24.935491
3 	0.003600 	0.646819 	28.172648
4 	1.510800 	0.582459 	21.252639
5 	0.439000 	0.533293 	18.977246
6 	0.117200 	0.486742 	18.672297
7 	0.249400 	0.419118 	16.091954
8 	0.443400 	0.387003 	14.778325
9 	0.003900 	0.380853 	14.919071
10 	0.003500 	0.371373 	13.488154


### Loading the dataset

```python
from datasets import DatasetDict, load_dataset, concatenate_datasets

atc0 = load_dataset("HF-SaLAI/salai_atc0", "base", token=hf_token)
atc0p2 = load_dataset("HF-SaLAI/salai_atc0", "part2", token=hf_token)
atc0p3 = load_dataset("HF-SaLAI/salai_atc0", "part3", token=hf_token)

dataset = DatasetDict()
dataset["train"] = concatenate_datasets(
    [atc0["train"], atc0p2["train"], atc0p3["train"]]).shuffle(seed=42)
shuffled_dataset = atc0["validation"].shuffle(seed=42)
dataset["validation"] = shuffled_dataset.select(range(500))

print(dataset)
```

### Creating a text normalizer

```python
import transformers.models.whisper.english_normalizer as en

english_text_normalizer = en.EnglishTextNormalizer({})
```

### Filtering the test dataset

Some examples will have an empty string after normalization, which will cause issues with the WER calculation. Here, we remove these examples.

```python
def is_transcript_empty(transcript):
    normalized_transcript = english_text_normalizer(transcript)
    return len(normalized_transcript) > 0

dataset["train"] = dataset["train"].filter(is_transcript_empty,
        input_columns=["text"])
dataset["validation"] = dataset["validation"].filter(is_transcript_empty,
        input_columns=["text"])
print(dataset)
```


### Creating a processor and its feature extractor and tokenizer

```python
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(model_name_or_path)
feature_extractor = processor.feature_extractor
tokenizer = processor.tokenizer
```

### Creating input features from audio data

```python
def prepare_dataset(batch):
    # compute log-Mel input features from input audio array
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"],
            sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(english_text_normalizer(
            batch["text"])).input_ids
    return batch

dataset = dataset.map(prepare_dataset,
                      remove_columns=dataset.column_names["train"],
                      num_proc=1)

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

    def __call__(self,
                 features: List[Dict[str, Union[List[int], torch.Tensor]]]) \
                        -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths
        # and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]}
                          for feature in features]
        batch = self.processor.feature_extractor.pad(input_features,
                                                     return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features,
                                                    return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().\
                            cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Note that we have the following issue when doing the training:
# he attention mask is not set and cannot be inferred from input because
# pad token is same as eos token.As a consequence, you may observe
# unexpected behavior. Please pass your input's `attention_mask` to obtain
# reliable results.
# The issue may be related to the padding of feature_extractor.
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
import torch

print(f"{torch.cuda.is_available() = }")
```

```python
from transformers import WhisperForConditionalGeneration

base_model = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path).to("cuda")
```

Override generation arguments - no tokens are forced as decoder outputs (see [`forced_decoder_ids`](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.forced_decoder_ids)), no tokens are suppressed during generation (see [`suppress_tokens`](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.suppress_tokens)):

```python
base_model.config.forced_decoder_ids = None
base_model.config.suppress_tokens = []
```


#### Apply LoRA


```python
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05, bias="none")

model = get_peft_model(base_model, config)

model.print_trainable_parameters()
```

#### Define the Training Configuration

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,          # change to a repo name of your choice
    per_device_train_batch_size=8,  # increase to 16 for larger datasets
    gradient_accumulation_steps=1,  # inc by 2x for every 2x dec in batch size
    learning_rate=1e-3,
    report_to="none",
    # warmup_steps=50,
    num_train_epochs=10,
    eval_strategy="epoch",
    fp16=True,
    per_device_eval_batch_size=1,
    generation_max_length=128,
    logging_steps=1,
    remove_unused_columns=False,  # required as the PeftModel forward doesn't
            # have the signature of the wrapped model's forward
    label_names=["labels"],       # same reason as above
    predict_with_generate=True,
    save_steps=0.1,               #if you wish to save checkpoints
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
model.config.use_cache = False  # silence warnings; re-enable for inference!
```

#### Train the adapter

```python
trainer.train()
```
```python
Running the trainer, we see the following message: "The attention mask is not set and cannot be inferred from input because pad token is same as eos token.As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."

It is normal according to https://www.reddit.com/r/KoboldAI/comments/yz26ol/how_to_fix_the_attention_mask_and_the_pad_token/.

We need to figure it out at a later time.
```

Epoch 	Training Loss 	Validation Loss 	Wer
1 	0.141900 	0.680747 	28.078818
2 	0.104900 	0.630345 	24.935491
3 	0.003600 	0.646819 	28.172648
4 	1.510800 	0.582459 	21.252639
5 	0.439000 	0.533293 	18.977246
6 	0.117200 	0.486742 	18.672297
7 	0.249400 	0.419118 	16.091954
8 	0.443400 	0.387003 	14.778325
9 	0.003900 	0.380853 	14.919071
10 	0.003500 	0.371373 	13.488154


0 	0.847700 	0.617785 	24.724373
2 	0.339100 	0.420697 	16.185785
4 	0.034800 	0.325710 	13.699273


Epoch 	Training Loss 	Validation Loss 	Wer
1 	0.537300 	0.593508 	19.382504
2 	0.139400 	0.497896 	17.724414
3 	0.180000 	0.436374 	15.666095
4 	0.007200 	0.431122 	14.351058
5 	0.010100 	0.415403 	12.349914


Epoch 	Training Loss 	Validation Loss 	Wer
1 	1.019400 	0.667351 	26.389866
2 	0.194500 	0.629990 	23.645320
3 	1.194200 	0.632762 	25.334272
4 	1.760800 	0.533477 	23.903354


Numbers used in the training:

-   Total number of train examples: 22490
-   Number of train examples in a batch: 8
-   Number of steps in 10 epoch: 28120
-   8 * 28120 = 224960 which is about 22490 * 10

### Saving the finetuned model locally and push to Hugging Face

```python
from transformers import WhisperForConditionalGeneration
from peft import PeftModel

# base_model = WhisperForConditionalGeneration.from_pretrained(
#              model_name_or_path, load_in_8bit=False, device_map="auto")
base_model = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path).to("cuda")
model = PeftModel.from_pretrained(base_model, adapter_to_choose)
print(f"{model = } \n\n")

# model.merge_and_unload() merges the adapter parameters with the base model
# parameters and unloads the adapter. This typically results in a standard
# model that can be used without needing the PEFT infrastructure.
model = model.merge_and_unload()
print(f"{model = } \n\n")
```

#### Saving locally

```python
model.save_pretrained(trained_model_local)
processor.save_pretrained(trained_model_local)
```

```python
print(trained_model_local)
```

#### Pushing to Hugging Face

```python
model.push_to_hub(trained_model_repo, safe_serialization=True)
```

```python
processor = WhisperProcessor.from_pretrained(trained_model_local)
processor.push_to_hub(trained_model_repo)
```

---

### Notes about model loading

Note that we can use two approaches for loading a model:

-   `model1 = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, load_in_8bit=False, device_map="auto")`
-   `model2 = WhisperForConditionalGeneration.from_pretrained(model_name_or_path).to("cuda")`

Here is a comparison of the above two approaches:

-   They both use the default 16-bit coefficient for the checkpoint.
-   The first one uses `device_map="auto"` to specify to automatically distributes model layers across available hardware, which can optimize performance and memory usage, especially in multi-GPU setups.
-   The second uses `.to("cuda")` to specify to move the entire model to a single GPU, which is straightforward but may not utilize multiple GPUs or balance resources as effectively.
-   The second approach is the preferred way for the single GPU case. If we use multiple GPUs, we can use the first one.
