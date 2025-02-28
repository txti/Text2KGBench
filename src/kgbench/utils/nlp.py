from pathlib import Path

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

_is_initialized: bool = False
_stemmer: PorterStemmer = PorterStemmer()

def _download_nltk_data():
    global _is_initialized

    if _is_initialized:
        return

    # Create the download directory if it doesn't exist
    download_dir = Path(nltk.data.path[0])
    download_dir.mkdir(parents=True, exist_ok=True)

    # Download the necessary NLTK data
    nltk.download("punkt", download_dir=download_dir, quiet=True)
    nltk.download("punkt_tab", download_dir=download_dir, quiet=True)

    _is_initialized = True

def stem(text: str):
    global _stemmer
    # Idempotent call to download NLTK data
    _download_nltk_data()

    # Return the stem of the text
    return _stemmer.stem(text)


def tokenize(text: str):
    # Idempotent call to download NLTK data
    _download_nltk_data()

    # Return a list of tokens
    return word_tokenize(text)
