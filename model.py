import pandas as pd
import numpy as np
import os 
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score
nltk.download('wordnet')

chunk_size=15000
batch_no=1
file_list=[]
for chunk in pd.read_csv('sms_with_category.csv',chunksize=chunk_size):
     chunk.to_csv('chunk '+str(batch_no)+'.csv',index=False)
     file_list.append('chunk '+str(batch_no)+'.csv')
     batch_no+=1

def listtostring(s):
    # initialize an empty string 
    str1 = " " 
    # return string   
    return (str1.join(s))

# object of naive bayes 
clf = MultinomialNB(alpha=0.0001, fit_prior=False)
for i in file_list:
	path= i
	data=pd.read_csv(path,lineterminator='\n')
	data=data.sample(frac=1)
	input_column='body'
	output_column='category'
	def preprocess_text_train(df,column):
        for i in range(len(df)):
            ######  REMOVING SPECIAL CHARACTERS
            df.loc[i,column]  = re.sub(r'\W',' ',str(df.loc[i,column]))
            # ######  REMOVING ALL SINGLE CHARACTERS
            df.loc[i,column]  = re.sub(r'\s+[a-zA-Z]\s+',' ',str(df.loc[i,column]))
            ######  REMOVING MULTIPLE SPACES WITH SINGLE SPACE
            df.loc[i,column]  = re.sub(r'\s+',' ',str(df.loc[i,column]))
            ##lower case 
            df.loc[i,column]=df.loc[i,column].lower()
            ps=PorterStemmer()
            l1=[]
            for word in df.loc[i,column].split():
                l1.append(ps.stem(word))
            df.loc[i,column]=listtostring(l1)
	    return df
	data=preprocess_text_train(data,input_column)
	data['category']=data['category'].fillna('promotional')
    data['category'].isnull().sum()
	data[input_column].head()
	x=data.loc[:,input_column]
	y=data.loc[:,output_column]
	############### USING BAG OF WORDS MODEL TO CONVERT FEATURES INTO NUMBERS ############
	tfidf_vect = TfidfVectorizer()
	X_counts   = tfidf_vect.fit_transform(x).toarray()
	#model train done 
	clf.fit(X_counts,y)

filename = 'model.pkl'
pickle.dump(clf, open(filename, 'wb'))
	