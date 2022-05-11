import pandas as pd
from IPython.display import display
import MoreThanSentiments as mts
my_dir_path = "E:/Package/Boilerplate/morethansentiments/Data/bbc/business"
df = mts.read_txt_files(PATH = my_dir_path)



df['sent_tok'] = df.text.apply(mts.sent_tok)
    
df['cleaned_data'] = pd.Series()    
for i in range(len(df['sent_tok'])):
    df['cleaned_data'][i] = [mts.clean_data(x,\
                                            lower = True,\
                                            punctuations=True,\
                                            number=False,\
                                            unicode=True,\
                                            stop_words=None) for x in df['sent_tok'][i]] 



df['Boilerplate'] = mts.Boilerplate(df.cleaned_data, n = 4, min_doc = 5)
df['Redundancy'] = mts.Redundancy(df.cleaned_data, n = 10)
df['Specificity'] = mts.Specificity(df.text)
df['Relative_prevalence'] = mts.Relative_prevalence(df.text)

display(df.head(3))