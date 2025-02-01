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
def Boilerplate(input_data: pd.Series, n: int = 4, min_doc: float = 5, max_doc: float = 0.75, get_ngram: bool = False):
    """
    Computes the boilerplate percentage in a document based on repeated n-grams.

    Parameters:
    - input_data (pd.Series): A Pandas Series where each element is a document (string).
    - n (int): The n-gram length (default=4).
    - min_doc (float): Minimum document occurrence for an n-gram to be considered (default=5).
    - max_doc (float): Maximum fraction of documents an n-gram can appear in (default=0.75).
    - get_ngram (bool): If True, returns the extracted n-grams and their counts instead of the boilerplate score.

    Returns:
    - If get_ngram=True: Returns a DataFrame with n-grams and their counts.
    - Otherwise: Returns a list containing the boilerplate score for each document.
    """

    # Ensure input_data is a Pandas Series
    if not isinstance(input_data, pd.Series):
        raise TypeError("input_data must be a Pandas Series.")

    # Convert all documents to strings (handle NaNs, numbers, etc.)
    input_data = input_data.astype(str).fillna("")

    doc_length = len(input_data)

    # Validate n-gram range
    assert 3 <= n <= 6, "Invalid value for n. Must be between 3 and 6."

    # Validate min_doc
    if 0 < min_doc < 1:
        min_doc = round(min_doc * doc_length)  # Convert percentage to count
    assert min_doc > 0, "min_doc must be greater than 0."
    
    # Validate max_doc
    if 0 < max_doc < 1:
        max_doc = round(max_doc * doc_length)
    assert 0 < max_doc <= doc_length, "max_doc must be between 0 and the total number of documents."

    # Extract n-grams per document
    ngram_list = []
    document_ngrams = []
    for doc in tqdm(input_data, desc="Extracting n-grams"):
        doc_ngrams = []
        for sent in nltk.sent_tokenize(doc):  # Sentence tokenize first
            sent_ngrams = list(nltk.ngrams(sent.split(), n))  # Generate n-grams
            doc_ngrams.extend(sent_ngrams)  # Store n-grams per document
        document_ngrams.append(doc_ngrams)
        ngram_list.extend(doc_ngrams)  # Flatten for global count

    # Count frequency of unique n-grams
    ngram_counts = pd.Series(ngram_list).value_counts().reset_index(name='counts')
    ngram_counts.columns = ['Ngrams', 'counts']

    if get_ngram:
        return ngram_counts  # Return n-gram frequency table

    # Filter n-grams based on min_doc and max_doc thresholds
    valid_ngrams = set(ngram_counts.query(f'counts >= {min_doc} and counts <= {max_doc}')['Ngrams'])

    # Compute boilerplate score
    boilerplate_scores = []
    for doc_ngrams, doc in tqdm(zip(document_ngrams, input_data), total=len(input_data), desc="Computing Boilerplate Score"):
        words_in_doc = len(doc.split())  # Total words in document
        flagged_words = sum(len(ng) for ng in doc_ngrams if ng in valid_ngrams)
        boilerplate_scores.append(flagged_words / words_in_doc if words_in_doc > 0 else 0)

    return boilerplate_scores


def Redundancy(input_data: pd.Series, n: int = 10):
    """
    Computes redundancy in a document based on repeated n-grams.

    Parameters:
    - input_data (pd.Series): A Pandas Series where each element is a document (string).
    - n (int): The n-gram length (default=10).

    Returns:
    - List of redundancy scores per document.
    """

    # Ensure input_data is a Pandas Series
    if not isinstance(input_data, pd.Series):
        raise TypeError("input_data must be a Pandas Series.")

    # Convert all documents to strings (handle NaNs, lists, numbers, etc.)
    input_data = input_data.astype(str).fillna("")

    doc_length = len(input_data)

    # Validate n-gram range
    assert 5 <= n <= 15, "Invalid value for n. Must be between 5 and 15."

    redundancy_scores = []
    
    # Extract n-grams and compute redundancy
    for doc in tqdm(input_data, desc="Extracting n-grams"):
        # Extract n-grams for the document
        ngram_list = list(nltk.ngrams(doc.split(), n))
        
        if not ngram_list:  # Handle empty documents
            redundancy_scores.append(0)
            continue

        # Count n-gram occurrences
        ngram_counts = pd.Series(ngram_list).value_counts()

        # Compute redundancy as % of n-grams that occur more than once
        repeated_ngrams = ngram_counts[ngram_counts > 1].sum()
        total_ngrams = len(ngram_list)
        
        redundancy_scores.append(repeated_ngrams / total_ngrams if total_ngrams > 0 else 0)

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
