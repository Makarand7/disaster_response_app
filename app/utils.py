import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def tokenize(text):
    """Tokenize and clean text."""
    stop_words = set(stopwords.words("english"))
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    clean_tokens = [w for w in words if w not in stop_words]
    return clean_tokens

