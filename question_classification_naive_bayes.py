import os
import string
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from nltk.tokenize import sent_tokenize, word_tokenize
#Load Excel file from the current directory
ques_file=pd.read_excel('Questions.xlsx',sheet_name=0)
ques_file.head()
#Seprate Type and Question from the data frame
label=ques_file[u'Type']
question=ques_file[u'Question']
tr_ques, te_ques, tr_label, te_label= model_selection.train_test_split(question, label, test_size=0.25, random_state=0)
vocab={}
'''Traverse through questions make a vocabolary with bag of word store the tokeninze text and along 
with there cooccurance.
''' 
for qu in tr_ques:
	list_word=[]
	#Tokenize and and split the inappropiate space from the text
	for clean_word in word_tokenize(str(qu).strip()):
		#change the text into lower case so as to reduce the dimentation of the data
		clean_word=clean_word.lower()
		if(len(clean_word)>2):
			if clean_word in vocab:
				vocab[clean_word]+=1
			else:
				vocab[clean_word]=1

num_words = [0 for i in range(max(vocab.values())+1)] 
freq = [i for i in range(max(vocab.values())+1)] 
for key in vocab:
    num_words[vocab[key]]+=1
cutoff_freq = 2
# For deciding cutoff frequency
num_words_above_cutoff = len(vocab)-sum(num_words[0:cutoff_freq]) 
#print("Number of words with frequency higher than cutoff frequency({}) :".format(cutoff_freq),num_words_above_cutoff)
features=[]
for key in vocab:
	if vocab[key]>=cutoff_freq:
		features.append(key)
#Initilize zero matrix for store tokenize text matrix

train_quest_data=np.zeros((len(tr_ques),len(features)))
inn=0
#print("QQ Word")
for sent in tr_ques:
	word_ls=[(word.strip()).lower() for word in sent.split()]
	#print(word_ls)
	for q_word in word_ls:
		if q_word in features:
			#print(q_word)
			train_quest_data[inn][features.index(q_word)]+=1
	inn=inn+1
tinn=0
test_quest_data=np.zeros((len(te_ques),len(features)))
for test_wo in te_ques:
	word_ls=[word.strip().lower() for word in test_wo.split()]
	for word in word_ls:
		if word in features:
			test_quest_data[tinn][features.index(word)]+=1
	tinn=tinn+1
#Instance Multinomial Naive Bias which calulate the probability of multiple classes

clf=MultinomialNB()
clf.fit(train_quest_data,tr_label)
#test_predi=clf.predict(test_quest_data)
teind=0
#print("Test Result")
for que in te_ques:
	teind=teind+1
#Argument for the Training Data

argumentList = sys.argv
test_quest=argumentList[1]
arg_test_quest_data=np.zeros((1,len(features)))
test_word=[t_word.strip().lower() for t_word in test_quest.split()]
for tw in test_word:
	if tw in features:
		arg_test_quest_data[0][features.index(tw)]+=1
test_predi=clf.predict(arg_test_quest_data)

print("\033[1;32;40m Classification Class   ")
print("\033[1;31;43m"+test_predi[0] + "\033[1;32;40m \033[1;32;40m")
