# MoreThanSentiments
Besides sentiment scores, this Python package offers various ways of quantifying text corpus based on multiple works of literature. Currently, we support the calculation of the following measures:

-   Boilerplate (Lang and Stice-Lawrence, 2015)
-   Redundancy (Cazier and Pfeiffer, 2015)
-   Specificity (Hope et al., 2016)
-   Relative_prevalence (Blankespoor, 2016)

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
                                                punctuations = True,\
                                                number = False,\
                                                unicode = True,\
                                                stop_words = False) for x in df['sent_tok'][i]] 
                                                
If you want to clean on the document level:

    df['cleaned_data'] = df.text.apply(mts.clean_data, args=(True, True, False, True, False))

For the data cleaning function, we offer the following options:
-   lower: make all the words to lowercase
-   punctuations: remove all the punctuations in the corpus
-   number: remove all the digits in the corpus
-   unicode: remove all the unicodes in the corpus
-   stop_words: remove the stopwords in the corpus

#### Boilerplate

    df['Boilerplate'] = mts.Boilerplate(sent_tok, n = 4, min_doc = 5, get_ngram = False)

Parameters:
-   input_data: this function requires tokenized documents.
-   n: number of the ngrams to use. The default is 4.
-   min_doc: when building the ngram list, ignore the ngrams that have a document frequency strictly lower than the given threshold. The default is 5 document. 30% of the number of the documents is recommended.
-   get_ngram: if this parameter is set to "True" it will return a datafram with all the ngrams and the corresponding frequency, and "min_doc" parameter will become ineffective.

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
