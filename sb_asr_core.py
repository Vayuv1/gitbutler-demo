import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import time
import difflib
from sb_ui import TranscriptUI
from speechbrain.inference.ASR import EncoderDecoderASR

# Configuration
SAMPLE_RATE = 16000
WINDOW_DURATION = 5.0
HOP_DURATION = 0.5
LOOKBACK_SIZE = int(SAMPLE_RATE * 2.0)  # 2s context
MIN_ENERGY = 0.01
MIN_WORD_COUNT = 2
SIMILARITY_THRESHOLD = 0.9

WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)
HOP_SIZE = int(SAMPLE_RATE * HOP_DURATION)

ui = TranscriptUI()
sliding_buffer = []
recent_lines = []
MAX_HISTORY = 5

asr_model = None

def load_model(name):
    global asr_model
    if name == "conformer":
        asr_model = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-transformer-transformerlm-librispeech",
            savedir="../pretrained_models/asr-transformer-transformerlm"
                    "-librispeech"
        )
    elif name == "wav2vec2":
        asr_model = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-wav2vec2-commonvoice-en",
            savedir="../pretrained_models/asr-wav2vec2-commonvoice-en"
        )
    else:
        raise ValueError("Unknown model type")

# ─── Utilities ─────────────────────────────────────────────────────

def list_devices():
    print("\nAvailable input devices:\n")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"[{i}] {dev['name']}")

def _find_pulseaudio_monitor_device():
    for i, dev in enumerate(sd.query_devices()):
        if "pulse" in dev["name"].lower() and dev["max_input_channels"] > 0:
            return i
    return None

def _normalize(text: str) -> str:
    """Lowercase and remove immediate duplicate words."""
    words = text.lower().split()
    cleaned = []
    for w in words:
        if not cleaned or w != cleaned[-1]:
            cleaned.append(w)
    return " ".join(cleaned)


def _is_similar(new: str, old: str) -> bool:
    """Check similarity ignoring case and duplicate words."""
    return (
        difflib.SequenceMatcher(None, _normalize(new), _normalize(old)).ratio()
        > SIMILARITY_THRESHOLD
    )

# ─── Transcription Core ───────────────────────────────────────────

def _process_window():
    global recent_lines
    if len(sliding_buffer) < WINDOW_SIZE:
        return

    window = sliding_buffer[-(WINDOW_SIZE + LOOKBACK_SIZE):]
    if np.max(window) < MIN_ENERGY:
        return

    chunk = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
    wav_lens = torch.tensor([1.0])

    try:
        prediction = asr_model.transcribe_batch(chunk, wav_lens=wav_lens)
        if isinstance(prediction[0], list):
            text = " ".join(prediction[0])
        else:
            text = prediction[0].strip()
        text = _normalize(text)

        if not text or len(text.split()) < MIN_WORD_COUNT:
            return

        if any(_is_similar(text, prev) for prev in recent_lines):
            return

        if recent_lines:
            last = recent_lines[-1].split()
            curr = text.split()
            overlap = 0
            for i in range(1, min(len(last), len(curr))):
                if last[-i:] == curr[:i]:
                    overlap = i
            if overlap > 0:
                text = " ".join(curr[overlap:])
                text = _normalize(text)

        ui.update_partial(text)
        ui.update_final(text)
        recent_lines.append(text)
        if len(recent_lines) > MAX_HISTORY:
            recent_lines.pop(0)

    except Exception as e:
        print(f"[error] {e}")

# ─── Input Modes ───────────────────────────────────────────────────

def transcribe_from_mic(device_index=None):
    print("Using microphone input...")
    def callback(indata, frames, time_info, status):
        sliding_buffer.extend(indata[:, 0].tolist())
        if len(sliding_buffer) > WINDOW_SIZE:
            del sliding_buffer[:len(sliding_buffer) - WINDOW_SIZE]
        ui.update_level(float(np.mean(np.abs(indata))))
        _process_window()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=HOP_SIZE,
        dtype="float32",
        device=device_index,
        callback=callback
    ):
        try:
            ui.run()
        except KeyboardInterrupt:
            print("\nTranscription stopped.")
            ui.stop()

def transcribe_from_system(device_index=None):
    print("Using system audio input...")
    if device_index is None:
        device_index = _find_pulseaudio_monitor_device()
    if device_index is None:
        print("PulseAudio monitor device not found.")
        return
    transcribe_from_mic(device_index)

def transcribe_from_usb(device_index):
    print(f"Using USB device index {device_index}")
    transcribe_from_mic(device_index)

def transcribe_from_file(filepath):
    print(f"Transcribing audio file: {filepath}")
    audio, sr = sf.read(filepath)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * SAMPLE_RATE / sr))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    def feed():
        for start in range(0, len(audio) - HOP_SIZE + 1, HOP_SIZE):
            chunk = audio[start:start + HOP_SIZE]
            sliding_buffer.extend(chunk.tolist())
            if len(sliding_buffer) > WINDOW_SIZE:
                del sliding_buffer[:len(sliding_buffer) - WINDOW_SIZE]
            ui.update_level(float(np.mean(np.abs(chunk))))
            _process_window()
            time.sleep(HOP_DURATION)

    import threading
    threading.Thread(target=feed, daemon=True).start()
    ui.run()
