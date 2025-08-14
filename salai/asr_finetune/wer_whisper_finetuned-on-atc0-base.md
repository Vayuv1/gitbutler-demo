# WER Evaluation with Whisper

## Using base medium-en model with the ATC0 dataset

### Using the HF token

```python
from dotenv import load_dotenv
load_dotenv()
```

### Loading the dataset

```python
from datasets import DatasetDict, load_dataset

atc0 = load_dataset("HF-SaLAI/salai_atc0", "base", use_auth_token=True) 

print(atc0)

audio_input = atc0["train"][1]["audio"]   # first decoded audio sample
transcription = atc0["train"][1]["text"]  # first transcription

print(audio_input)
print(transcription)

dataset = DatasetDict()
dataset["test"] = atc0["test"]

print(dataset)
```

### Loading the Whisper model---the base model without finetuning

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Input model
# To force a reload, run "rm -rf ~/.cache/huggingface/transformers/*"
# model_name_or_path = "openai/whisper-medium.en"
model_name_or_path = "./whisper-lora-atc0-base2/whisper-medium.en-finetuned-on-atc0-base2"

processor = WhisperProcessor.from_pretrained(model_name_or_path)
model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path).to("cuda")
```

### Filtering the test dataset

Some examples will have an empty string after normalization, which will cause issues with the WER calculation. Here, we remove these examples.

```python
def is_transcript_empty(transcript):
    normalized_transcript = processor.tokenizer._normalize(transcript)
    return len(normalized_transcript) > 0

dataset["test"] = dataset["test"].filter(is_transcript_empty,
                                         input_columns=["text"],
                                        )
print(dataset)
```

### Transcribing the test data 

Note that using the print out functions, we can see we have the following format for the transripts: 
```
batch['text'] = 'continental three twenty five descend and maintain five thousand\n'
transcription = '<|startoftranscript|><|notimestamps|> Now 325, center maintain 5000<|endoftext|>'
 batch['reference'] = 'continental 325 descend and maintain 5000'
batch['prediction'] = 'now 325 center maintain 5000'
batch['text'] = "that's for continental three twenty three down to five thousand\n"
transcription = '<|startoftranscript|><|notimestamps|> As for CNO 323 down to 5,000<|endoftext|>'
 batch['reference'] = 'that is for continental 323 down to 5000'
batch['prediction'] = 'as for cno 323 down to 5000'
```

```python
import json
import torch

def map_to_pred(batch):
    audio = batch["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    batch["reference"] = processor.tokenizer._normalize(batch['text'])
    
    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("cuda"))[0]
    transcription = processor.decode(predicted_ids)

    batch["prediction"] = processor.tokenizer._normalize(transcription)

    # print(f"{batch['text'] = }")
    # print(f"{transcription = }")
    # print(f"{ batch['reference'] = }")
    # print(f"{batch['prediction'] = }")
    
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

file_path = "wer_whisper_finetuned-on-atc0-base_atc0.json"

with open(file_path, 'w') as json_file:
    json.dump(asr_result, json_file, indent=4)
```

### Checking the result

```python
import json

file_path = "wer_whisper_finetuned-on-atc0-base_atc0.json"
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

#: 16.492751203495004
```

```python
Issue:

Hallucination kills the WER. Need to address it. 
```
