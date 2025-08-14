# Loading dataset from HF

The code embedded in this markdown file is supposed to run as cells in a Jupyter notebook. This can be done using Jupyter Lab.

## Loading the ATC0 dataset

### Using the HF token

```python
from dotenv import load_dotenv
load_dotenv()
```

### Loading the `base` configuration

```python
from datasets import load_dataset

atc0 = load_dataset("HF-SaLAI/salai_atc0", "base", use_auth_token=True)

print(atc0)

audio_input = atc0["train"][1]["audio"]   # first decoded audio sample
transcription = atc0["train"][1]["text"]  # first transcription

print(audio_input)
print(transcription)
```

### Loading the `part2` and `part3` configurations

```python
atc02 = load_dataset("HF-SaLAI/salai_atc0", "part2", use_auth_token=True)

print(atc02)
# load audio sample on the fly
audio_input = atc02["train"][0]["audio"]   # first decoded audio sample
transcription = atc02["train"][0]["text"]  # first transcription

print(audio_input)
print(transcription)
```

```python
atc03 = load_dataset("HF-SaLAI/salai_atc0", "part3", use_auth_token=True)

print(atc03)

audio_input = atc03["train"][0]["audio"]   # first decoded audio sample
transcription = atc03["train"][0]["text"]  # first transcription

print(audio_input)
print(transcription)
```

```python

```
