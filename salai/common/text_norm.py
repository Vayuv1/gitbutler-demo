# salai/asr_finetune/common/text_norm.py
"""
This script provides a consistent text normalization function to be used
across different ASR engines (Whisper, SpeechBrain, etc.) to ensure
fair WER comparisons.
"""
import re

def normalize_transcript(transcript: str) -> str:
    """
    Applies the standard text normalization required by the project.

    Normalization steps:
    1. Convert to uppercase.
    2. Keep digits.
    3. Keep only specified punctuation: '.' and '/'.
    4. Collapse multiple whitespace characters into a single space.

    Args:
        transcript: The raw transcript string.

    Returns:
        The normalized transcript string.
    """
    if not isinstance(transcript, str):
        return ""

    # 1. Convert to uppercase
    text = transcript.upper()

    # 3. Keep only allowed characters: A-Z, 0-9, space, '.', and '/'
    # The regex '[^A-Z0-9 .\/]' matches any character that is NOT in the set.
    text = re.sub(r'[^A-Z0-9 .\/]', '', text)

    # 4. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

