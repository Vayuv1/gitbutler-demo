import subprocess
from transformers import (
    AutoProcessor, AutoModelForSpeechSeq2Seq, GenerationConfig)

# Load and convert the model to CTranslate2 format
def convert_model():
    my_model = "HF-SaLAI/whisper-medium-atc0.en"
    processor = AutoProcessor.from_pretrained(my_model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(my_model)

    # Set the custom generation parameters in a GenerationConfig file
    # See https://github.com/huggingface/transformers/blob/main/src/
    # transformers/models/whisper/configuration_whisper.py
    generation_config = GenerationConfig(
        max_length=448,
        suppress_tokens=[
            1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62,
            63, 90, 91, 92, 93, 357, 366, 438, 532, 685, 705, 796, 930, 1058,
            1220, 1267, 1279, 1303, 1343, 1377, 1391, 1635, 1782, 1875, 2162,
            2361, 2488, 3467, 4008, 4211, 4600, 4808, 5299, 5855, 6329, 7203,
            9609, 9959, 10563, 10786, 11420, 11709, 11907, 13163, 13697, 13700,
            14808, 15306, 16410, 16791, 17992, 19203, 19510, 20724, 22305,
            22935, 27007, 30109, 30420, 33409, 34949, 40283, 40493, 40549,
            47282, 49146, 50257, 50357, 50358, 50359, 50360, 50361],
        begin_suppress_tokens=[220, 50256]
    )
    model.config.update(generation_config.to_dict())

    # Save the model and processor locally
    model.save_pretrained("whisper-medium-atc.en")
    processor.save_pretrained("whisper-medium-atc.en")

    # Convert the model
    # ct2-transformers-converter --model whisper-medium-atc.en
    # --output_dir whisper-medium-atc-ct2.en --quantization bfloat16

    command = [  # Define the command as a list of arguments
        "ct2-transformers-converter",
        "--model", "whisper-medium-atc.en",
        "--output_dir", "whisper-medium-atc-ct2.en",
        "--quantization", "bfloat16"
    ]

    subprocess.run(command, check=True)     # Run the command

    # For the options of quantization, see
    # https://opennmt.net/CTranslate2/quantization.html
    # A good choice is int8_bfloat16, which means:
    # -  int8: The model weights are quantized to 8-bit integers
    # -  bfloat16: The computations (e.g., matrix multiplications, activations)
    #    are performed in bfloat16 precision instead of full float32.
    # This hybrid approach aims to save memory (via int8 weights) and maintain
    # numerical stability (via bfloat16 compute).

# Convert the model first
convert_model()
