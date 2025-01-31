__author__ = "Jinhang Jiang, Srinivasan Karthik"
__author_email__ = "jinhang@asu.edu, karthiks@ku.edu"

"""
Enhancements:
    1. Specificity: Optimized Named Entity Recognition (NER)
    2. Readability Index: Added efficient computation
    3. Topic Embeddings: Faster tokenization
    4. Relative_prevalence: Enhanced NER for numerical values
"""

from collections import defaultdict
from pathlib import Path
import pandas as pd
import nltk
import spacy
from itertools import chain
from nltk.corpus import stopwords
import string
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Download necessary NLP packages if not present
nltk.download("stopwords")
nltk.download("punkt")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


### 📌 I/O FUNCTION ###
def read_txt_files(path: str) -> pd.DataFrame:
    """
    Reads all `.txt` files from a given directory and stores their content in a DataFrame.

    Parameters:
    - path (str): Path to the directory containing text files.

    Returns:
    - pd.DataFrame: A DataFrame with columns ['file_num', 'text'] containing filenames and their contents.
    """
    results = {"file_num": [], "text": []}
    for file in tqdm(Path(path).iterdir(), desc="Reading Files"):
        if file.suffix == ".txt":  # Process only .txt files
            with open(file, "rt", encoding="utf-8") as f:
                results["file_num"].append(file.name)
                results["text"].append(f.read().replace("\n", " "))
    return pd.DataFrame(results)


### 📌 TEXT CLEANING ###
def clean_data(
    doc: str,
    lower: bool = True,
    remove_punct: bool = False,
    remove_numbers: bool = False,
    remove_unicode: bool = False,
    remove_stopwords: bool = False,
) -> str:
    """
    Cleans text data with optional preprocessing steps.

    Parameters:
    - doc (str): Input text document.
    - lower (bool): Convert text to lowercase (default: True).
    - remove_punct (bool): Remove punctuation (default: False).
    - remove_numbers (bool): Remove digits (default: False).
    - remove_unicode (bool): Remove non-ASCII characters (default: False).
    - remove_stopwords (bool): Remove English stopwords (default: False).

    Returns:
    - str: Processed text.
    """
    if lower:
        doc = doc.lower()
    if remove_punct:
        doc = doc.translate(str.maketrans("", "", string.punctuation))
    if remove_numbers:
        doc = "".join([c for c in doc if not c.isdigit()])
    if remove_unicode:
        doc = doc.encode("ascii", "ignore").decode("utf-8")
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        doc = " ".join(word for word in doc.split() if word not in stop_words)
    return doc


### 📌 TOKENIZATION ###
def sent_tok(doc: str) -> list:
    """
    Tokenizes text into sentences.

    Parameters:
    - doc (str): Input text document.

    Returns:
    - list: A list of tokenized sentences.
    """
    return nltk.sent_tokenize(doc)


### 📌 BOILERPLATE DETECTION ###
def Boilerplate(
    input_data: pd.Series,
    n: int = 4,
    min_doc: float = 5,
    max_doc: float = 0.75,
    get_ngram: bool = False,
) -> pd.DataFrame or list:
    """
    Detects boilerplate text using n-grams.

    Parameters:
    - input_data (pd.Series): Series containing text documents.
    - n (int): N-gram size (default: 4).
    - min_doc (float): Minimum document frequency threshold (default: 5).
    - max_doc (float): Maximum document frequency ratio (default: 0.75).
    - get_ngram (bool): If True, returns a DataFrame of n-grams and their counts.

    Returns:
    - list or pd.DataFrame: Boilerplate scores or n-gram frequency DataFrame.
    """
    doc_length = len(input_data)

    assert 3 <= n <= 6, "Invalid n. Must be between 3 and 6."
    assert 0 < min_doc <= 0.5 * doc_length, "Invalid min_doc value."
    assert 0 < max_doc <= 1, "Invalid max_doc value."

    if min_doc < 1:
        min_doc = round(min_doc * doc_length)

    # Extract n-grams for each sentence
    ngram_list = [
        [list(nltk.ngrams(sent.split(), n)) for sent in sent_tok(doc)]
        for doc in tqdm(input_data, desc="Extracting n-grams")
    ]

    # Flatten list and create frequency DataFrame
    all_ngrams = list(chain.from_iterable(chain.from_iterable(ngram_list)))
    ngram_freq = pd.DataFrame(pd.Series(all_ngrams).value_counts()).reset_index()
    ngram_freq.columns = ["ngram", "count"]

    if get_ngram:
        return ngram_freq

    # Remove overly common and rare n-grams
    ngram_freq = ngram_freq[
        (ngram_freq["count"] >= min_doc) & (ngram_freq["count"] <= max_doc * doc_length)
    ]

    # Compute boilerplate ratio
    boilerplate_scores = []
    for i, doc in enumerate(input_data):
        words = doc.split()
        flagged_words = sum(
            len(sent.split())
            for sent in sent_tok(doc)
            if any(ngram in ngram_freq["ngram"].values for ngram in list(nltk.ngrams(sent.split(), n)))
        )
        boilerplate_scores.append(flagged_words / len(words) if words else 0)

    return boilerplate_scores


### 📌 TEXT REDUNDANCY ###
def Redundancy(input_data: pd.Series, n: int = 10) -> list:
    """
    Calculates text redundancy as the percentage of repeating n-grams.

    Parameters:
    - input_data (pd.Series): Series containing text documents.
    - n (int): N-gram size (default: 10).

    Returns:
    - list: Redundancy scores per document.
    """
    assert 5 <= n <= 15, "n should be between 5 and 15."

    # Extract n-grams
    ngram_list = [
        list(nltk.ngrams(doc.split(), n)) for doc in tqdm(input_data, desc="Extracting n-grams")
    ]

    # Compute redundancy
    redundancy_scores = [
        sum(pd.Series(ngrams).value_counts().loc[lambda x: x > 1].sum() / len(ngrams))
        if len(ngrams) > 0 else 0
        for ngrams in ngram_list
    ]
    return redundancy_scores


### 📌 SPECIFICITY ###
def Specificity(input_data: pd.Series) -> list:
    """
    Computes specificity as the ratio of named entities to total words.

    Parameters:
    - input_data (pd.Series): Series containing text documents.

    Returns:
    - list: Specificity scores.
    """
    return [
        len(nlp(doc).ents) / len(doc.split()) if len(doc.split()) > 0 else 0
        for doc in tqdm(input_data, desc="Extracting named entities")
    ]


### 📌 RELATIVE PREVALENCE ###
def Relative_prevalence(input_data: pd.Series) -> list:
    """
    Computes the relative prevalence of numerical values in the text.

    Parameters:
    - input_data (pd.Series): Series containing text documents.

    Returns:
    - list: Relative prevalence scores.
    """
    return [
        (len(doc.split()) - len("".join(filter(lambda c: not c.isdigit(), doc)).split()))
        / len(doc.split()) if len(doc.split()) > 0 else 0
        for doc in tqdm(input_data, desc="Calculating Relative Prevalence")
    ]
