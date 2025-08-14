import re

def normalize_atc_text(text: str) -> str:
    """Simple ATC friendly normalization.
    Uppercase letters. Keep digits. Collapse spaces. Keep period and slash.
    Remove other punctuation.
    """
    if text is None:
        return ""
    t = text.upper()
    # Replace any punctuation except period and slash with space
    t = re.sub(r"[^A-Z0-9\./ ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t