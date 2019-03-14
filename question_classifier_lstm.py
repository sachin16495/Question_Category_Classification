import tensorflow as tf
import sys
from keras.utils import np_utils
import numpy as np
from sklearn import model_selection
from scipy.spatial.distance import cdist
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.layers import LSTM,Dense,GRU,Embedding,SpatialDropout1D
import pandas as pd
#Load Excel File
ques_file=pd.read_excel('Questions.xlsx',sheet_name=0)
print(ques_file.head)
label=ques_file[u'Type']
question=ques_file[u'Question']
tr_ques, te_ques, tr_label, te_label= model_selection.train_test_split(question, label, test_size=0.10,shuffle=False)
num_words=100
#Tokenize the text
tokenize=Tokenizer(num_words=num_words)
tokenize.fit_on_texts(tr_ques)
idx=tokenize.word_index
x_train_token=tokenize.texts_to_sequences(tr_ques)
x_test_token=tokenize.texts_to_sequences(te_ques)

#Find max tokens 
num_tokens=[len(token) for token in x_train_token+x_test_token]
num_tokens=np.array(num_tokens)
max_tokens=np.mean(num_tokens)+2*np.std(num_tokens)
max_tokens=int(max_tokens)
print("Max Tokens")
print(max_tokens)
#print(tr_label)
argumentList = sys.argv

#Classes
classes={u'Education':0,u'Nutrition':1,u'Health':2,u'Child Psychology':3}
classesa=['Education','Nutrition','Health','Child Psychology']

#One Hot encoding of labels
tr_label_cod=[classes[leb] for leb in tr_label]
y_train = np_utils.to_categorical(tr_label_cod, 4)

#Adding padding to the training data
pad='pre'
x_train_pad=pad_sequences(x_train_token,maxlen=max_tokens,padding=pad,truncating=pad)
x_test_pad=pad_sequences(x_test_token,maxlen=max_tokens,padding=pad,truncating=pad)

#embedding_size=8
#emb_dim = 16

reuse_word = 200
#Model Architecture
model = Sequential()
model.add(Embedding(reuse_word, 64, input_length=max_tokens))
model.add(LSTM(64, dropout=0.2))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())
if len(argumentList)<3:
	argumentList.append('test')
if argumentList[2]=='train':
	model.fit(x_train_pad, y_train, epochs=100, batch_size=5,validation_split=0.25)
	model.save_weights("model200.h5")
model.load_weights('model200.h5')

#txt = ["Which are best CBSE/ICSE schools in Wakad Pune ?"]
test_quest=argumentList[1]
seq = tokenize.texts_to_sequences(test_quest)
x_test_pad=pad_sequences(seq,maxlen=max_tokens,padding=pad,truncating=pad)
pred = model.predict(x_test_pad)
# Get the maximum element from a Numpy array
#maxElement = [i for i in range(0,len()) if pred[i]==np.amax(pred[0])]
#print(pred[0])
ind = np.unravel_index(np.argmax(pred[0], axis=None), pred[0].shape)
#print(ind[0])
#print(test_quest)
print("---------------------------------------------------------")
print("\033[1;32;40m Classification Class  \033[0m 1;32;40m")
#print(" \033[0;37;42m"+classesa[ind[0]]+ "\033[0;37;42m")
print("\033[1;31;43m"+classesa[ind[0]] + "\033[1;32;40m \033[1;32;40m")
print(pred[0])

