import pandas as pd
from IPython.display import display
import morethansentiments as mts
my_dir_path = "E:/Package/Boilerplate/morethansentiments/Data/bbc/business"
df = mts.read_txt_files(PATH = my_dir_path)



temp_col = []
for i in range(len(df)):
    temp_col.append(mts.sent_tok(df.text[i]))
    
for i in range(len(temp_col)):
    temp_col[i] = [mts.clean_data(x, lower = True, punctuations=True, number=False, unicode=True, stop_words=None) for x in temp_col[i]]
    
df['cleaned_data'] = pd.Series(temp_col)    



df['Boilerplate'] = mts.Boilerplate(df.cleaned_data, n = 4, min_doc = 5)
df['Redundancy'] = mts.Redundancy(df.cleaned_data, n = 10)
df['Specificity'] = mts.Specificity(df.text)
df['Relative_prevalence'] = mts.Relative_prevalence(df.text)

display(df.head(3))