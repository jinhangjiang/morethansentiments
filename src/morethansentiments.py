__author__='Jinhang Jiang, Srinivasan Karthik'
__author_email__='jinhang@asu.edu, karthiks@ku.edu'

'''
To be added:
    1. Specificity: GPT-3 NER
    2. Readability Index: 4 ways + average option
    3. Topic Embeddings
    4. Relative_prevalence: add NER for money value
'''

from collections import defaultdict
from pathlib import Path
import pandas as pd
import nltk
import spacy.cli
spacy.cli.download("en_core_web_sm")
from itertools import chain
from nltk.corpus import stopwords
import string
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")




def test_func():
    return ("Lib is ready")



def read_txt_files(PATH:str):
    
    results = defaultdict(list)
    for file in Path(PATH).iterdir():
        with open(file, "rt",newline='') as file_open:
            results["file_num"].append(file.name)
            results["text"].append(file_open.read().replace('\n'," "))
    df = pd.DataFrame(results)
    
    return df


def clean_data(doc:str, lower=True, punctuations=None, number=None,
               unicode=None, stop_words:str=None):
    if lower == True:
        doc = doc.lower() #convert all the letters to lower
        
    if punctuations == True:
        doc = doc.translate(str.maketrans('', '', string.punctuation)) #remove punctuations
    
    if number == True:
        doc = ''.join([i for i in doc if not i.isdigit()])  #remove digits
    
    if unicode == True:
        doc = doc.encode('ascii', 'ignore').decode("utf-8")  #remove unicode
        
    if stop_words != None:
        nltk.download('stopwords')
        stop = stopwords.words(stop_words)
        doc = doc.apply(lambda x: ' '.join(x for x in x.split() if x not in stop)) #remove stop words
        
    
    return doc


def sent_tok(doc:str):
    
    return nltk.sent_tokenize(doc)


def Boilerplate(input_data: pd.Series, n: int = 4, min_doc: int = 5):
    '''
    #### LOGIC (Lang and Stice-Lawrence, 2015):
    
    # Collect all tetragrams (ordered group of four words within each sentence) in each document in a list
    
    # Remove frequently used tetragrams: Remove tetragrams that occur 75% or more across documents from the list.  
    
    # Identify tetragrams that occur in at least 30% of the documents or an average of at least 5 times -
        per document in the list (phrases commonly used in financial disclosures). Discard other tetragrams from the list. 
        
    # BOILERPLATE = % of total words in document that are in sentences containing boilerplate tetragrams. 
    '''
    try:
        assert 3 <= n <= 6
        
    except AssertionError:
        
        "Invalid Value for n (int) [3,6]"
        
        raise
    
    try:
        
        assert 0 <= min_doc <= len(input_data)/2
        
    except AssertionError:
        
        "Invalid Value for min_doc (int) [0, len(input_data)/2]"
        
        raise
        
    # capture the 4-grams for each sentence for all the documents
    ngram = [0]*len(input_data)
    for i in tqdm(range(len(input_data)), desc = 'Get the Boilerplate'):
        
        ngram[i] = [0]*len(input_data[i])
        for j in range(len(input_data[i])):
            if input_data[i] !='':
                ngram[i][j] = list(nltk.ngrams(input_data[i][j].split(), n))
    
    # get all unique 4-grams per document to one list
    list_all_ngrams = list(chain(*[set(list(chain(*ngram[i]))) for i in range(len(ngram))]))
    
    
    # remove the 4-grams that appeared in less than 5 documents [CHECKHERE, percentage is not included]
    fndf = pd.DataFrame(pd.DataFrame({'Ngrams':list_all_ngrams}).\
                     value_counts().\
                     rename_axis('unique_ngrams').\
                     reset_index(name='counts').\
                     query(f'counts >= {min_doc}'))

    # NWoS, calculate the number of the words in each sentence per document, and store them
    temp_nwos = [0]*len(input_data)
    for i in tqdm(range(len(temp_nwos)), desc = "Get the Length of Sentence"):
        temp_nwos[i] = [len(j.split()) for j in input_data[i]]    
    
    # Flag the sentence
    sent_flag = [[]]*len(input_data)
    for i in tqdm(range(len(sent_flag)), desc = 'Flag the Sentence'):

        sent_flag[i] = [0]*len(ngram[i])

        for j in range(len(ngram[i])):

            if any(x in ngram[i][j] for x in fndf.unique_ngrams):

                sent_flag[i][j] = temp_nwos[i][j]

            else:

                sent_flag[i][j] = 0    
    
    #Final Calculation of Boilerplate
    display("======================== Boilerplate Calculation Started =========================")
    boilerplate = [sum(sent_flag[i])/sum(temp_nwos[i]) for i in range(len(input_data))]
    display("======================== Boilerplate Calculation Finished ========================")
    
    return boilerplate





def Redundancy(cleaned_data: pd.Series, n: int = 10):
    '''
    #  % of 10-grams that occur more than once in each document (Cazier and Pfeiffer, 2015)
    '''
    try:
        assert 5 <= n <= 15
        
    except AssertionError:
        
        "Invalid Value for n (int) [5,15]"
        
        raise
        
    # capture the 10-grams for each sentence for all the documents
    ngram = [0]*len(cleaned_data)
    for i in tqdm(range(len(cleaned_data)), desc = 'Get the Redundancy'):
        
        ngram[i] = [0]*len(cleaned_data[i])
        for j in range(len(cleaned_data[i])):
            if cleaned_data[i] !='':
                ngram[i][j] = list(nltk.ngrams(cleaned_data[i][j].split(),10))
    
    # get all 10-grams per document
    list_ngrams_per_doc = [list(chain(*ngram[i])) for i in range(len(ngram))]
    
    # calculate the Redundancy
    redundancy = [pd.DataFrame({"Ten_grams":list_ngrams_per_doc[i]}).\
                  value_counts().\
                  loc[lambda x: x>1].\
                  sum()/\
                  len(list_ngrams_per_doc[i]) for i in range(len(list_ngrams_per_doc))]
    
    return redundancy



def Specificity(data: pd.Series):
    '''
    #### LOGIC (Hope et al., 2016):
    
    # Extract named entities. 
    
    # Specificity is the no. of specific entity names, quantitative values, times/dates 
    
    # All scaled by the total number of words in document.
    '''
    
    
    ner = spacy.load('en_core_web_sm')
    
    specificity = [0]*len(data)
    
    for i in tqdm(range(len(data)), desc = 'Get the Specificity'):
        specificity[i] = len(ner(data[i]).ents)/len(data[i])
    
    #[len(ner(data[i]).ents)/len(data[i]) for i in tqdm(range(len(data)))]
    
    return specificity




def Relative_prevalence(data:pd.Series):
    
    ''' (Blankespoor, 2016)
    # relative prevalence of informative numbers in the text or “hard” information 
    '''
    
    #[CHECKHERE, Money and Number entities are not included]
    
#     nlp = spacy.load("en_core_web_md", disable = ["tagger","parser"])

#     df = pd.DataFrame({"Text":["this is a text about Germany","this is another about Trump"]})

#     texts = df["Text"].to_list()
#     ents = []
#     for doc in nlp.pipe(texts):
#         for ent in doc.ents:
#             if ent.label_ == "GPE":
#                 ents.append(ent)
    
    relative_prevalence = [0]*len(data)
    
    for i in tqdm(range(len(data)), desc = 'Get the Relative_prevalence'):
        
        a = len(data[i].split())
        doc = ''.join([j for j in data[i] if not j.isdigit()])
        b = len(doc.split())
        relative_prevalence[i] = (a-b)/a
        
    return relative_prevalence