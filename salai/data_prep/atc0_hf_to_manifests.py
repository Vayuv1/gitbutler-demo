# salai/asr_finetune/data_prep/atc0_hf_to_manifests.py
"""
This script downloads the ATC0 dataset from Hugging Face, converts the audio
to 16 kHz mono WAV files, and creates CSV manifest files for training,
validation, and testing.
"""
import os
import argparse
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
import torchaudio
from speechbrain.dataio.dataio import read_audio_info


def process_atc0_dataset(
        output_dir: Path,
        wav_output_dir: Path,
        dataset_name: str,
        hf_token: str,
        config: str,
        train_fraction: float,
        validation_samples: int,
):
    """
    Downloads, processes, and creates manifests for the ATC0 dataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_output_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset
    print(f"Loading dataset '{dataset_name}' with config '{config}'...")
    atc0 = load_dataset(dataset_name, config, use_auth_token=hf_token,
                        trust_remote_code=True)

    # Process splits
    for split in atc0.keys():
        print(f"Processing split: {split}...")

        # Create specific output directory for wav files of the current split
        split_wav_dir = wav_output_dir / config / f"{split}_wav"
        split_wav_dir.mkdir(parents=True, exist_ok=True)

        records = []
        # Use enumerate to generate a unique index for each example
        for i, example in enumerate(
                tqdm(atc0[split], desc=f"Converting {split} audio")):

            # Generate a unique, deterministic ID
            file_id = f"{config}_{split}_{i:08d}"

            audio_path = split_wav_dir / f"{file_id}.wav"

            # Save audio to WAV if it doesn't exist
            if not audio_path.exists():
                audio_array = example["audio"]["array"]
                sample_rate = example["audio"]["sampling_rate"]

                # Convert to tensor, resample, and save
                audio_tensor = torch.tensor(audio_array).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=16000)
                resampled_audio = resampler(audio_tensor)
                torchaudio.save(audio_path, resampled_audio, 16000)

            # Get audio info
            info = read_audio_info(str(audio_path))

            records.append({
                # CORRECTED: Use 'ID' (uppercase) as required by SpeechBrain
                "ID": file_id,
                "duration": info.num_frames / info.sample_rate,
                "wav": str(audio_path.resolve()),
                "transcript": example["text"],
            })

        # Create and save manifest
        manifest_df = pd.DataFrame(records)
        manifest_path = output_dir / f"{config}_{split}.csv"
        manifest_df.to_csv(manifest_path, index=False)
        print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare ATC0 dataset manifests.")

    # Updated arguments to match the README and provide flexibility
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the manifest CSV files.")
    parser.add_argument("--wav_output_dir", type=str, required=True,
                        help="Directory to save the converted WAV files.")
    parser.add_argument("--dataset_name", type=str,
                        default="HF-SaLAI/salai_atc0",
                        help="Name of the Hugging Face dataset.")
    parser.add_argument("--hf_token", type=str, required=True,
                        help="Your Hugging Face API token for private datasets.")
    parser.add_argument("--config", type=str, default="base",
                        choices=["base", "part2", "part3"],
                        help="Dataset configuration to process.")
    parser.add_argument("--train_fraction", type=float, default=1.0,
                        help="Fraction of training data to use.")
    parser.add_argument("--validation_samples", type=int, default=None,
                        help="Number of validation samples to use.")

    args = parser.parse_args()

    process_atc0_dataset(
        output_dir=Path(args.output_dir),
        wav_output_dir=Path(args.wav_output_dir),
        dataset_name=args.dataset_name,
        hf_token=args.hf_token,
        config=args.config,
        train_fraction=args.train_fraction,
        validation_samples=args.validation_samples,
    )
