[![License](https://img.shields.io/badge/License-BSD_3--Clause-green.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI](https://img.shields.io/pypi/v/morethansentiments)](https://pypi.org/project/morethansentiments/)
[![Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://codeocean.com/capsule/7670045/tree)
[![Downloads](https://pepy.tech/badge/morethansentiments)](https://pepy.tech/project/morethansentiments)

# MoreThanSentiments
Besides sentiment scores, this Python package offers various ways of quantifying text corpus based on multiple works of literature. Currently, we support the calculation of the following measures:

-   Boilerplate (Lang and Stice-Lawrence, 2015)
-   Redundancy (Cazier and Pfeiffer, 2015)
-   Specificity (Hope et al., 2016)
-   Relative_prevalence (Blankespoor, 2016)

A medium blog is here: [MoreThanSentiments: A Python Library for Text Quantification](https://towardsdatascience.com/morethansentiments-a-python-library-for-text-quantification-e57ff9d51cd5)

## Citation

If this package was helpful in your work, feel free to cite it as

- Jiang, Jinhang, and Karthik Srinivasan. "MoreThanSentiments: A text analysis package." Software Impacts 15 (2023): 100456. https://doi.org/10.1016/J.SIMPA.2022.100456

## Installation

The easiest way to install the toolbox is via pip (pip3 in some
distributions):

    pip install MoreThanSentiments
    

## Usage

#### Import the Package

    import MoreThanSentiments as mts
    
#### Read data from txt files

    my_dir_path = "D:/YourDataFolder"
    df = mts.read_txt_files(PATH = my_dir_path)
    
#### Sentence Token

    df['sent_tok'] = df.text.apply(mts.sent_tok)
    
#### Clean Data

If you want to clean on the sentence level:

    df['cleaned_data'] = pd.Series()
    for i in range(len(df['sent_tok'])):
        df['cleaned_data'][i] = [mts.clean_data(x,\
                                                lower = True,\
                                                remove_punct=True,\
                                                remove_numbers=False,\
                                                remove_unicode=True,\
                                                remove_stopwords=False) for x in df['sent_tok'][i]]
                                                
If you want to clean on the document level:

    df['cleaned_data'] = df.text.apply(mts.clean_data, args=(True, True, False, True, False))

For the data cleaning function, we offer the following options:
-   lower: make all the words to lowercase
-   remove_punct: remove all the punctuations in the corpus
-   remove_numbers: remove all the digits in the corpus
-   remove_unicode: remove all the unicodes in the corpus
-   remove_stopwords: remove the stopwords in the corpus

#### Boilerplate

    df['Boilerplate'] = mts.Boilerplate(df.cleaned_data, n = 4, min_doc = 5, get_ngram = False)

Parameters:
-   input_data: this function requires tokenized documents.
-   n: number of the ngrams to use. The default is 4.
-   min_doc: when building the ngram list, ignore the ngrams that have a document frequency strictly lower than the given threshold. The default is 5 document. 30% of the number of the documents is recommended.
-   get_ngram: if this parameter is set to "True" it will return a datafram with all the ngrams and the corresponding frequency, and "min_doc" parameter will become ineffective.
-   max_doc: when building the ngram list, ignore the ngrams that have a document frequency strictly lower than the given threshold. The default is 75% of document. It can be percentage or integer.

#### Redundancy

    df['Redundancy'] = mts.Redundancy(df.cleaned_data, n = 10)
    
Parameters:
-   input_data: this function requires tokenized documents.
-   n: number of the ngrams to use. The default is 10.

#### Specificity

    df['Specificity'] = mts.Specificity(df.text)

Parameters:
-   input_data: this function requires the documents without tokenization

#### Relative_prevalence

    df['Relative_prevalence'] = mts.Relative_prevalence(df.text)
    
Parameters:
-   input_data: this function requires the documents without tokenization


For the full code script, you may check here:
-   [Script](https://github.com/jinhangjiang/morethansentiments/blob/main/tests/test_code.py)
-   [Jupyter Notebook](https://github.com/jinhangjiang/morethansentiments/blob/main/Boilerplate.ipynb)
