import re
import unicodedata

def clean_text(text: str) -> str:
    """Normalize Unicode and whitespace without removing meaningful punctuation."""
    text = unicodedata.normalize("NFC", text).replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(re.sub(r"[\t\f\v ]+", " ", line).strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()
