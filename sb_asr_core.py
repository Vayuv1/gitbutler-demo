"""SpeechBrain-based real-time ASR utilities."""

import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import time
import difflib
from sb_ui import TranscriptUI
from speechbrain.inference.ASR import EncoderDecoderASR

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
# Audio is processed in overlapping windows. HOP_DURATION controls how often a
# new chunk is sent to the model while WINDOW_DURATION is the overall length of
# audio considered each time `_process_window` runs.
SAMPLE_RATE = 16000
WINDOW_DURATION = 5.0
HOP_DURATION = 0.5

# When forming a window for inference we include a short lookback so the model
# has context for the start of the new chunk.
LOOKBACK_SIZE = int(SAMPLE_RATE * 2.0)  # 2 seconds of context

# Heuristics to filter noise and extremely short outputs
MIN_ENERGY = 0.01
MIN_WORD_COUNT = 2
SIMILARITY_THRESHOLD = 0.9

WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)
HOP_SIZE = int(SAMPLE_RATE * HOP_DURATION)

# ---------------------------------------------------------------------------
# Globals used across functions
# ---------------------------------------------------------------------------
ui = TranscriptUI()             # Handles terminal display
sliding_buffer: list[float] = []  # Raw audio from the input device
recent_lines: list[str] = []      # Keeps prior partial texts to avoid repeats
MAX_HISTORY = 5

asr_model: EncoderDecoderASR | None = None


def load_model(name: str) -> None:
    """Load a SpeechBrain ASR model by name."""
    global asr_model
    if name == "conformer":
        asr_model = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-transformer-transformerlm-librispeech",
            savedir="../pretrained_models/asr-transformer-transformerlm"
                    "-librispeech",
        )
    elif name == "wav2vec2":
        asr_model = EncoderDecoderASR.from_hparams(
            source="speechbrain/asr-wav2vec2-commonvoice-en",
            savedir="../pretrained_models/asr-wav2vec2-commonvoice-en",
        )
    else:
        raise ValueError("Unknown model type")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def list_devices() -> None:
    """Print all available input devices."""
    print("\nAvailable input devices:\n")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"[{i}] {dev['name']}")


def _find_pulseaudio_monitor_device() -> int | None:
    """Attempt to locate a PulseAudio 'monitor' input for system audio."""
    for i, dev in enumerate(sd.query_devices()):
        if "pulse" in dev["name"].lower() and dev["max_input_channels"] > 0:
            return i
    return None


def _normalize(text: str) -> str:
    """Normalize text by lowercasing and removing duplicated words."""
    words = text.lower().split()
    cleaned: list[str] = []
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


# ---------------------------------------------------------------------------
# Core transcription pipeline
# ---------------------------------------------------------------------------

def _process_window() -> None:
    """Run ASR on the latest audio window and update the UI."""
    global recent_lines

    # Only run inference when we have enough audio gathered
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

        # Ignore extremely short or empty outputs
        if not text or len(text.split()) < MIN_WORD_COUNT:
            return

        # Skip output if it's too similar to a recent line
        if any(_is_similar(text, prev) for prev in recent_lines):
            return

        # Remove overlapping words with previous prediction
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

    except Exception as e:  # noqa: BLE001
        print(f"[error] {e}")


# ---------------------------------------------------------------------------
# Input modes
# ---------------------------------------------------------------------------

def transcribe_from_mic(device_index: int | None = None) -> None:
    """Transcribe audio coming from a microphone device."""

    print("Using microphone input...")

    def callback(indata, frames, time_info, status):  # noqa: D401
        """Collect audio and feed the ASR window."""
        sliding_buffer.extend(indata[:, 0].tolist())
        if len(sliding_buffer) > WINDOW_SIZE:
            del sliding_buffer[: len(sliding_buffer) - WINDOW_SIZE]
        ui.update_level(float(np.mean(np.abs(indata))))
        _process_window()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=HOP_SIZE,
        dtype="float32",
        device=device_index,
        callback=callback,
    ):
        try:
            ui.run()
        except KeyboardInterrupt:
            print("\nTranscription stopped.")
            ui.stop()


def transcribe_from_system(device_index: int | None = None) -> None:
    """Transcribe system audio via PulseAudio monitor."""

    print("Using system audio input...")
    if device_index is None:
        device_index = _find_pulseaudio_monitor_device()
    if device_index is None:
        print("PulseAudio monitor device not found.")
        return
    transcribe_from_mic(device_index)


def transcribe_from_usb(device_index: int) -> None:
    """Transcribe from an external USB audio device."""

    print(f"Using USB device index {device_index}")
    transcribe_from_mic(device_index)


def transcribe_from_file(filepath: str) -> None:
    """Feed audio from a file to the recognizer in real time."""

    print(f"Transcribing audio file: {filepath}")
    audio, sr = sf.read(filepath)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample

        audio = resample(audio, int(len(audio) * SAMPLE_RATE / sr))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    def feed() -> None:
        for start in range(0, len(audio) - HOP_SIZE + 1, HOP_SIZE):
            chunk = audio[start : start + HOP_SIZE]
            sliding_buffer.extend(chunk.tolist())
            if len(sliding_buffer) > WINDOW_SIZE:
                del sliding_buffer[: len(sliding_buffer) - WINDOW_SIZE]
            ui.update_level(float(np.mean(np.abs(chunk))))
            _process_window()
            time.sleep(HOP_DURATION)

    import threading

    threading.Thread(target=feed, daemon=True).start()
    ui.run()
