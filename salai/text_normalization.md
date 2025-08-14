# WER Evaluation with Whisper

The code embedded in this markdown file is supposed to run as cells in a Jupyter notebook. This can be done using Jupyter Lab.

## Using base medium-en model with the ATC0 dataset

This is just to experiment with the text normalizer of Whisper.

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

### Loading the Whisper processor

```python
from transformers import WhisperProcessor

# Input model
model_name_or_path = "openai/whisper-medium.en"

processor = WhisperProcessor.from_pretrained(model_name_or_path)
```

### Loading the Whisper text normanlizer

```python
import inspect
import transformers.models.whisper.english_normalizer as en

# List all classes in the module
classes = inspect.getmembers(en, inspect.isclass)
for a_class in classes:
    print(f"{a_class = }")

# a_class = ('BasicTextNormalizer', <class 'transformers.models.whisper.english_normalizer.BasicTextNormalizer'>)
# a_class = ('EnglishNumberNormalizer', <class 'transformers.models.whisper.english_normalizer.EnglishNumberNormalizer'>)
# a_class = ('EnglishSpellingNormalizer', <class 'transformers.models.whisper.english_normalizer.EnglishSpellingNormalizer'>)
# a_class = ('EnglishTextNormalizer', <class 'transformers.models.whisper.english_normalizer.EnglishTextNormalizer'>)
# a_class = ('Fraction', <class 'fractions.Fraction'>)
```

```python
processor_normalizer = processor.tokenizer._normalize
basic_text_normalizer_nosplit = en.BasicTextNormalizer()
english_text_normalizer = en.EnglishTextNormalizer({})
```

```python
i = 0
for text in dataset["test"]["text"]:
    print(f"                               {text = }")
    print(f"{basic_text_normalizer_nosplit(text) = }")
    print(f"{         processor_normalizer(text) = }")
    print(f"{      english_text_normalizer(text) = }\n\n")
    i = i + 1
    if i >= 100:
        break
```
