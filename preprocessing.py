"""Text preprocessing shared by the training script and the Flask app.

Keeping this in ONE place avoids train/serve skew (the model must see text
cleaned exactly the same way at prediction time as it did during training).
"""
import re
import string

_PUNCT_RE = re.compile("[%s]" % re.escape(string.punctuation))


def wordpre(text):
    """Clean a raw article string. Non-strings (e.g. NaN) become ''."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)                # [bracketed] text
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # URLs (must run BEFORE \W removal)
    text = re.sub(r"<.*?>+", "", text)                 # html tags
    text = re.sub(r"\W", " ", text)                    # special characters
    text = _PUNCT_RE.sub("", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)               # words containing digits
    return text
