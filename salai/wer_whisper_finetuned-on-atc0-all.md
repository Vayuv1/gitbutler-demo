# WER Evaluation with Whisper

## Using base medium-en model with the ATC0 dataset

### Using the HF token

```python
import os
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
```

### Parameters

```python
# the base model name or path
model_name_or_path = "openai/whisper-medium.en"
output_dir = 'whisper-lora-atc0-all'
adapter_to_choose = f"{output_dir}/checkpoint-28120"
trained_model_name = f"{output_dir}/whisper-medium.en-finetuned-on-atc0-all"
model_name_or_path = "./whisper-lora-atc0-all/whisper-medium.en-finetuned-on-atc0-all"
```

### Loading the dataset

```python
from datasets import DatasetDict, load_dataset

atc0 = load_dataset("HF-SaLAI/salai_atc0", "base", token=hf_token)

print(atc0)

audio_input = atc0["train"][1]["audio"]   # first decoded audio sample
transcription = atc0["train"][1]["text"]  # first transcription

print(audio_input)
print(transcription)

dataset = DatasetDict()
dataset["test"] = atc0["test"]

print(dataset)
```

### Creating a text normalizer

```python
import transformers.models.whisper.english_normalizer as en

english_text_normalizer = en.EnglishTextNormalizer({})
```

### Loading the Whisper model finetuned on the entire ATC0 dataset

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Input model
processor = WhisperProcessor.from_pretrained(model_name_or_path)
model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path).to("cuda")
```

### Filtering the test dataset

Some examples will have an empty string after normalization, which will cause issues with the WER calculation. Here, we remove these examples.

```python
def is_transcript_empty(transcript):
    normalized_transcript = english_text_normalizer(transcript)
    return len(normalized_transcript) > 0

dataset["test"] = dataset["test"].filter(is_transcript_empty,
                                         input_columns=["text"],
                                        )
print(dataset)
```

### Transcribing the test data

Note that using the print out functions, we can see we have the following format for the transripts:

```
      batch['text'] = 'washington ah u_s_air six eighty six is ah with you level ten thousand\n'
      transcription = '<|startoftranscript|><|notimestamps|>washington ah u s air 686 is ah with you level 10000<|endoftext|>'
 batch['reference'] = 'washington ah u s air 686 is ah with you level 10000'
batch['prediction'] = 'washington ah u s air 686 is ah with you level 10000'
      batch['text'] = 'u_s_air six eighty six proceed direct to washington\n'
      transcription = '<|startoftranscript|><|notimestamps|>u s air 686 proceed direct to washington<|endoftext|>'
 batch['reference'] = 'u s air 686 proceed direct to washington'
batch['prediction'] = 'u s air 686 proceed direct to washington'
```

```python
import json
import torch

def map_to_pred(batch):
    audio = batch["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    batch["reference"] = english_text_normalizer(batch['text'])

    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("cuda"))[0]
    transcription = processor.decode(predicted_ids)

    batch["prediction"] = english_text_normalizer(transcription)

    print(f"{batch['text'] = }")
    print(f"{transcription = }")
    print(f"{ batch['reference'] = }")
    print(f"{batch['prediction'] = }")

    return batch

# result = dataset["validation"].map(map_to_pred)
result = dataset["test"].map(map_to_pred)
```

### Converting the ASR results to a dict of lists

```python
refs = list(result["reference"])
asrs = list(result["prediction"])

asr_result = {"ref": refs, "asr": asrs}
print(f"{len(refs) = }")
```

### Saving the results to a JSON file

```python
import json

file_path = "wer_whisper_finetuned-on-atc0-all.json"

with open(file_path, 'w') as json_file:
    json.dump(asr_result, json_file, indent=4)
```

### Checking the result

```python
import json

file_path = "wer_whisper_finetuned-on-atc0-all.json"
with open(file_path, 'r') as json_file:
    asr_result = json.load(json_file)

for i in range(20):
    print(f'{asr_result["ref"][i] = }')
    print(f'{asr_result["asr"][i] = }')
```

### Calculating the WER

```python
from evaluate import load

wer = load("wer")
print(100 * wer.compute(references=asr_result["ref"],
                        predictions=asr_result["asr"]))

#: 14.130283551771155
# 15.401953418482345
```

```python

```
