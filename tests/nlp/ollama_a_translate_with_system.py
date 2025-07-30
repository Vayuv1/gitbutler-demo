import argparse
import json
import requests

# Ollama API endpoint
url = "http://localhost:11434/api/generate"

def get_cli_args():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Get the LLM model and ATC transcript.")
    
    # Define arguments
    parser.add_argument(
        "llm_model", type=str, 
        help="An LLM model (gemma3:4b, gemma3:1b, qwen3:1.7b)"
    )
    parser.add_argument("transcript", type=str, help="Transcript text string")
    
    # Parse arguments
    args = parser.parse_args()
    # Assign variables
    llm_model = args.llm_model
    transcript = args.transcript
    
    return llm_model, transcript

# text = "cessna one two three descend and maintain five thousand turn left heading two seven zero"
# text = "cessna one two three turn left heading two seven zero"

system_prompt = (
    "You are an ATC communication extort. "
    "Translate the prompt text to a single sentence with casing, punctuation, " 
    "and correct numerical numbers. " 
    "Use output JSON key 'text'. "
    "Below are examples of translations: "
    "1. 'riddle one three five climb and maintain eleven thousand' to 'Riddle 135, climb and maintain 11,000.' "
    "2. 'eclipse one three five alpha turn right heading two three four.' to 'Eclipse 135A, turn right heading 234.' "
    "3. 'daytona tower eclipse one three five alpha leaving nine thousand for seven thosand.' to 'Daytona Tower, Eclipse 135A, leaving 9,000 for 7,000.' "
)

# print(prompt)
# OLLAMA_MODEL = "gemma3:4b"
# OLLAMA_MODEL = "gemma3:1b"
# OLLAMA_MODEL = "qwen3:1.7b"

def translate_to_normal(llm_model: str = "gemma3:1b", transcript: str = None): 
# Request data
    data = {
        "model": llm_model,
        "system": system_prompt,
        "prompt": transcript, 
        "format": "json",
        "stream": False
    }

    # Send POST request
    response = requests.post(url, json=data)

    # Print the response
    if response.status_code == 200:
        normalized_data = json.loads(response.json()["response"])
        normalized_transcript = next(iter(normalized_data.values()))
        return normalized_transcript
    else:
        return "None"

if __name__ == "__main__":
    llm_model, raw_transcript = get_cli_args()
    nrm_transcript = translate_to_normal(llm_model, raw_transcript)
    # print(f"{transcript} => {next(iter(normalized_text.values()))}")
    print(f"\nRaw transcript:        {raw_transcript}")
    print(f"Normalized transcript: {nrm_transcript}")
