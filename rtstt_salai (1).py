import os
import torch
import logging
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time
import queue # Import the queue module
from RealtimeSTT import AudioToTextRecorder
from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, HSplit
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
import textwrap

# Audio meter
class AudioMeter(TextArea):
    def __init__(self, **kwargs):
        super().__init__(read_only=True, multiline=False, wrap_lines=False, height=1, **kwargs)
        self.text = ""
        self.max_width = 78 # Adjust based on your terminal width
        self._app = None # Initialize _app attribute

    @property
    def app(self):
        # This getter ensures we return the application instance if it's set
        return self._app

    @app.setter
    def app(self, value):
        # This setter allows the application to be assigned to the widget
        self._app = value

    def update_level(self, level: float):
        # level should be between 0.0 and 1.0
        num_bars = int(level * self.max_width)
        self.text = "▮" * num_bars
        # Invalidate the application to force a redraw
        if self.app: # Check if app is set before invalidating
            self.app.invalidate()

# UI Setup
style = Style.from_dict({
    "header": "bold underline",
    "partial": "fg:cyan",
    "assembled": "fg:green",
    "meter": "fg:yellow bg:black", # Style for the audio meter
})

partial_output = TextArea(style="class:partial", scrollbar=True, multiline=True, wrap_lines=True)
final_output = TextArea(style="class:assembled", scrollbar=True, multiline=True, wrap_lines=True)
audio_meter = AudioMeter(style="class:meter") # Instantiate the new audio meter

layout = Layout(HSplit([
    TextArea(text="\U0001F539 Partial Transcript Updates", style="class:header", height=1),
    partial_output,
    TextArea(text="\U0001F538 Final Transcript Paragraph", style="class:header", height=1),
    final_output,
    TextArea(text="\U0001F4C9 Audio Input Level", style="class:header", height=1), # Header for meter
    audio_meter, # Add the audio meter to the layout
]))

kb = KeyBindings()
@kb.add("c-c")
def _(event):
    audiodata_dir = "audiodata"
    os.makedirs(audiodata_dir, exist_ok=True)
    output_path = os.path.join(audiodata_dir, "transcript.txt")
    wrapped_text = textwrap.fill(final_output.text, width=80)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(wrapped_text)
    recorder.shutdown()
    # Ensure the audio meter stream is stopped if it's running
    if 'audio_stream' in globals() and globals()['audio_stream'] is not None:
        globals()['audio_stream'].stop()
        globals()['audio_stream'].close()
    event.app.exit()


final_words = []

import re

partial_history = []              
last_partial_display = "• ..."    # Default fallback on startup

def handle_partial(text: str):
    global partial_history, last_partial_display

    text = text.strip()
    
    if not text or text.lower() == "you":
        # If silence and screen is blank, restore last display
        if not partial_output.text.strip():
            partial_output.text = last_partial_display
            partial_output.buffer.cursor_position = len(last_partial_display)
            if partial_output.app:
                partial_output.app.invalidate()
        return

    # Extract last sentence-like fragment
    sentences = re.split(r'(?<=[.!?])\s+', text)
    last = sentences[-1].strip()
    if not last:
        return

    bullet = f"• {last}"

    # Avoid repeating last point
    if partial_history and partial_history[-1] == bullet:
        return

    # Add new point
    partial_history.append(bullet)
    if len(partial_history) > 20:
        partial_history = partial_history[-20:]

    updated_text = "\n".join(partial_history)

    # Update only if changed
    if updated_text != partial_output.text:
        partial_output.text = updated_text
        partial_output.buffer.cursor_position = len(updated_text)
        last_partial_display = updated_text  # Fallback in silence

        if partial_output.app:
            partial_output.app.invalidate()

# To vaoid repetations

def find_best_attachment_point(existing_words, new_words):
    """
    Finds the best index in existing_words to attach the new_words.
    This helps handle overlaps and corrections.
    """
    if not existing_words or not new_words:
        return len(existing_words)

    # Search for the longest prefix of new_words that exists near the end of existing_words
    for len_to_test in range(len(new_words), 0, -1):
        prefix_to_find = new_words[:len_to_test]
        
        # Search within a reasonable window at the end of the existing words
        search_window = existing_words[-(len(new_words) + 10):]
        for i in range(len(search_window)):
            if search_window[i:i+len_to_test] == prefix_to_find:
                # Return the index in the original `existing_words` list
                return len(existing_words) - len(search_window) + i
    
    return len(existing_words) # If no overlap, append to the end

def handle_final(text: str):
    """
    This new function handles stabilized text. It can backtrack and correct the
    final transcript based on new, more accurate outputs from the model.
    """
    global final_words
    text = text.strip()
    if not text or text.lower() == "you":
        return

    new_words = text.split()

    # Find the optimal point to attach or correct the transcript
    attachment_index = find_best_attachment_point(final_words, new_words)
    
    # Replace the words from the attachment point onwards with the new, corrected words
    final_words = final_words[:attachment_index] + new_words
    
    # Update the display
    final_output.text = " ".join(final_words)
    final_output.buffer.cursor_position = len(final_output.text)
    
    # Ensure UI redraw
    if final_output.app:
        final_output.app.invalidate()




def find_pulseaudio_monitor_device():
    print("Scanning for PulseAudio input devices...")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        # Look for "monitor" in the device name and ensure it has input channels
        if "pulse" in device["name"].lower() and "monitor" in device["name"].lower() and device["max_input_channels"] > 0:
            print(f"Found PulseAudio Monitor: {device['name']} at index {i}")
            return i
    print("No PulseAudio monitor device found directly.")
    print("Searching for any PulseAudio input device...")
    for i, device in enumerate(devices):
        if "pulse" in device["name"].lower() and device["max_input_channels"] > 0:
            print(f"Found generic PulseAudio Input: {device['name']} at index {i}")
            return i
    return None

# --- Audio Metering Thread Function ---
def audio_level_thread(q: queue.Queue, device_index: int, samplerate: int = 16000, blocksize: int = 1024):
    print(f"Starting audio level monitoring on device index {device_index}")
    try:
        def callback(indata, frames, time_info, status): # Renamed 'time' to 'time_info' to avoid conflict with imported time module
            if status:
                logging.warning(status)
            level = np.mean(np.abs(indata))
            try:
                q.put_nowait(level) # Use put_nowait to avoid blocking
            except queue.Full:
                pass # Queue is full, skip this frame
            
        with sd.InputStream(device=device_index, samplerate=samplerate, channels=1, dtype='float32', blocksize=blocksize, callback=callback) as stream:
            globals()['audio_stream'] = stream # Store stream globally to stop it later
            print("Audio meter stream started. Press Ctrl+C to exit.")
            while stream.active:
                time.sleep(0.1) # Keep the thread alive while the stream is active
    except Exception as e:
        print(f"Error in audio level thread: {e}")
        # If the device is busy, or not found, try to use default
        try:
            print("Attempting to use default input device for meter...")
            with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32', blocksize=blocksize, callback=callback) as stream:
                globals()['audio_stream'] = stream
                print("Audio meter stream started on default device. Press Ctrl+C to exit.")
                while stream.active:
                    time.sleep(0.1)
        except Exception as e_default:
            print(f"Could not start audio meter stream on any device: {e_default}")
    finally:
        print("Audio meter stream stopped.")


def transcribe_from_microphone(compute_type):
    device_index = sd.default.device[0]
    print(f"Using default microphone index {device_index}")
    run_transcription(device_index, compute_type=compute_type)

def transcribe_from_system(compute_type, device_index=None):
    if device_index is None:
        device_index = find_pulseaudio_monitor_device()

    if device_index is None:
        print("No suitable PulseAudio input found. Please ensure PulseAudio is running and a monitor device exists, or specify a device index with --device.")
        return

    device_name = sd.query_devices()[device_index]['name']
    print(f"Using PulseAudio device index {device_index}: {device_name}")
    run_transcription(device_index=device_index, compute_type=compute_type)

def transcribe_from_usb(device_index, compute_type):
    print(f"Using USB input device index {device_index}")
    run_transcription(device_index, compute_type=compute_type)

def transcribe_from_file(file_path, compute_type):
    print(f"Transcribing file with chunk feeding: {file_path}")
    target_samplerate = 16000
    chunk_duration = 0.5
    chunk_size = int(target_samplerate * chunk_duration)

    try:
        with sf.SoundFile(file_path, "r") as f:
            audio = f.read(dtype='float32')
            input_samplerate = f.samplerate
            if input_samplerate != target_samplerate:
                print(f"Resampling from {input_samplerate} Hz to {target_samplerate} Hz")
                from scipy.signal import resample
                audio = resample(audio, int(len(audio) * target_samplerate / input_samplerate))
    except FileNotFoundError:
        print(f"Audio file not found: {file_path}")
        return
    except Exception as e:
        print(f"Error processing file: {e}")
        return

    def feed_chunks():
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            if len(chunk) == 0:
                break
            recorder.feed_audio(chunk)
            # Update audio meter with the level of the fed chunk
            level = np.mean(np.abs(chunk))
            audio_meter.update_level(level)
            time.sleep(chunk_duration) # Simulate real-time feeding
        print("File feeding complete.")
        # Optionally, reset meter after file processing
        audio_meter.update_level(0.0)

    run_transcription(device_index=None, audio_data=None, compute_type=compute_type)
    # Start the file feeding in a separate thread
    threading.Thread(target=feed_chunks, daemon=True).start()

def run_transcription(device_index=None, audio_data=None, compute_type="float16"):
    global recorder, app # Make app global so we can invalidate it

    # Create a queue for audio levels
    audio_level_q = queue.Queue(maxsize=1) # Small queue to only hold the latest level

    # Start the audio meter update thread
    def update_meter_from_queue():
        while True:
            try:
                level = audio_level_q.get(timeout=0.1) # Get level with a small timeout
                audio_meter.update_level(level)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error updating meter from queue: {e}")
                break

    # Start the thread that reads from the queue and updates the meter UI
    meter_ui_updater_thread = threading.Thread(target=update_meter_from_queue, daemon=True)
    meter_ui_updater_thread.start()

    # Start the separate audio capture thread for the meter (only if using live input)
    if audio_data is None: # Only start the meter capture for live audio input
        # Determine the device index for the audio meter
        meter_device_index = device_index
        if meter_device_index is None: # If no specific device is given for transcription, try default
            try:
                meter_device_index = sd.default.device[0]
                print(f"No specific device index provided for meter. Using default input device: {sd.query_devices()[meter_device_index]['name']}")
            except Exception as e:
                print(f"Could not determine default input device for audio meter: {e}")
                meter_device_index = None # Fail gracefully if no default device
        
        if meter_device_index is not None:
            meter_capture_thread = threading.Thread(target=audio_level_thread, args=(audio_level_q, meter_device_index), daemon=True)
            meter_capture_thread.start()
        else:
            print("Skipping audio meter capture as no valid input device could be determined.")

    recorder = AudioToTextRecorder(
        use_microphone=(audio_data is None),
        input_device_index=device_index, # This will be used by RealtimeSTT
        sample_rate=16000,
        language="en",
        compute_type=compute_type,
        enable_realtime_transcription=True,
        realtime_model_type="tiny.en",
        realtime_processing_pause=0.5,
        spinner=False,
        on_realtime_transcription_update=handle_partial,
        on_realtime_transcription_stabilized=handle_final,
        level=logging.WARNING,
    )
    recorder.start()

    # Create the prompt_toolkit application
    app = Application(layout=layout, key_bindings=kb, full_screen=True, style=style)

    # **Crucially, set the 'app' attribute for all TextAreas after the Application is created**
    audio_meter.app = app
    partial_output.app = app
    final_output.app = app

    app.run()
