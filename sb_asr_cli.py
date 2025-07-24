import argparse
from sb_asr_core import (
    load_model,
    transcribe_from_mic,
    transcribe_from_file,
    transcribe_from_system,
    transcribe_from_usb,
    list_devices
)

def main():
    parser = argparse.ArgumentParser(description="Real-Time ASR CLI Tool")
    parser.add_argument("--mode", choices=["mic", "system", "usb", "file"], required=True)
    parser.add_argument("--device", type=int, help="Input device index (for mic/system/usb)")
    parser.add_argument("--path", type=str, help="Path to audio file (if mode=file)")
    parser.add_argument("--model", choices=["conformer", "wav2vec2"], default="conformer",
                        help="Choose ASR model: 'conformer' or 'wav2vec2'")
    parser.add_argument("--list-devices", action="store_true", help="List input devices and exit")

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    load_model(args.model)

    if args.mode == "mic":
        transcribe_from_mic(device_index=args.device)

    elif args.mode == "system":
        transcribe_from_system(device_index=args.device)

    elif args.mode == "usb":
        if args.device is None:
            print("Please specify --device for USB mode.")
            return
        transcribe_from_usb(device_index=args.device)

    elif args.mode == "file":
        if not args.path:
            print("Please specify --path for file mode.")
            return
        transcribe_from_file(filepath=args.path)

if __name__ == "__main__":
    main()
