import argparse
import sounddevice as sd
from rtstt_salai import (
    transcribe_from_microphone,
    transcribe_from_system,
    transcribe_from_usb,
    transcribe_from_file,
)

def list_input_devices():
    print("\nAvailable Input Devices:")
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"[{idx}] {dev['name']}")

def main():
    parser = argparse.ArgumentParser(description="RealTimeSTT CLI Tool")
    appaa = parser.add_argument

    appaa("--mode", choices=["mic", "system", "usb", "file"],
          help="Select input mode: mic, system, usb, file")
    appaa("--device", type=int,
          help="Manual device index override (for USB/system)")
    appaa("--path", type=str,
          help="Path to audio file (for file mode)")
    appaa("--compute_type", choices=["float16", "float32", "int8", "int8_float16", "int8_float32"],
          default="float16", help="Whisper model compute type")
    appaa("--list-devices", action="store_true",
          help="List available input devices and exit")

    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    if args.mode == "mic":
        transcribe_from_microphone(args.compute_type)

    elif args.mode == "system":
        transcribe_from_system(args.compute_type, device_index=args.device)

    elif args.mode == "usb":
        if args.device is None:
            raise ValueError("Please specify --device for USB mode.")
        transcribe_from_usb(args.device, args.compute_type)

    elif args.mode == "file":
        if args.path is None:
            raise ValueError(" Please specify --path for file mode.")
        transcribe_from_file(args.path, args.compute_type)

    else:
        raise ValueError(" Please provide a valid --mode option.")

if __name__ == "__main__":
    main()
