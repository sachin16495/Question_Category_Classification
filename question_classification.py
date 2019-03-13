import os
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from nltk.tokenize import sent_tokenize, word_tokenize
ques_file=pd.read_excel('Questions.xlsx',sheet_name=0)#,index_col=0,columns={"Type","Question"} ,axis='columns')
print("Training Data Overview")
ques_file.head()
label=ques_file[u'Type']
question=ques_file[u'Question']
tr_ques, te_ques, tr_label, te_label= model_selection.train_test_split(question, label, test_size=0.25, random_state=0)
print(len(tr_ques))
print("Vocabulary Building")
#Vocabulary Building 
vocab={}
#print(tr_ques[][0:])
print(tr_ques[4])
for indx in range(len(tr_ques)):
	list_word=[]
	#[indx].split())
	print(indx)
	for clean_word in word_tokenize(tr_ques[indx].strip()):
		#clean_word=word.strip(string.pantuation).lower()
		#print((clean_word))
		if(len(clean_word)>2):
			if clean_word in vocab:
				vocab[clean_word]+=1
			else:
				vocab[clean_word]=1
cutoff_freq = 80
# For deciding cutoff frequency
num_words_above_cutoff = len(vocab)-sum(num_words[0:cutoff_freq]) 
print("Number of words with frequency higher than cutoff frequency({}) :".format(cutoff_freq),num_words_above_cutoff)

