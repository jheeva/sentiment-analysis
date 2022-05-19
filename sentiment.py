# -*- coding: utf-8 -*-
"""
Created on Mon May 16 23:36:32 2022

@author: End User
"""

import pandas as pd
import os
import re


import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

PATH_LOGS=os.path.join(os.getcwd(),'logs')
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'model.h5')

#step 1)loading data
url='https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'



df=pd.read_csv(url)
review=df['review']
review_dummy=review.copy()   #x_train

sentiment=df['sentiment']
sentiment_dummy=sentiment.copy()  #y_train

#step 2)data inspection

review_dummy[3]
sentiment_dummy[3]

review_dummy[11]
sentiment_dummy[11]

#step 3)data cleaning


#to remove html tags <br>
for index,text in enumerate(review_dummy):
    review_dummy[index]=re.sub('<.*?>','',text)
    
#to convert into lowercase and split it and to remove numerrical text
for index,text in enumerate(review_dummy):
    review_dummy[index]=re.sub('[a-zA-Z]',' ',text).lower().split()
    
#step 4)feature selection
#step 5)data preprocessing
#data visualization for reviews
    
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

num_words=10000
oov_token='oov'

tokenizer=Tokenizer(num_words=num_words,oov_token=oov_token)
tokenizer.fit_on_texts(review_dummy)

#%%to save the tokenizer for the deployment purpose
TOKENIZER_JSON_PATH=os.path.join(os.getcwd(), 'tokenizer_data.json')
token_json=tokenizer.to_json()

import json

with open(TOKENIZER_JSON_PATH,'w')as json_file:
    json.dump(token_json,json_file)
    





#to observe the no of words
word_index=tokenizer.word_index
print(word_index)
print(dict(list(word_index.items())[0:10]))   

#vectorize the sequences of text
review_dummy=tokenizer.texts_to_sequences(review_dummy)


temp=[np.shape(i)for i in review_dummy]
np.mean(temp)
review_dummy=pad_sequences(review_dummy,maxlen=200,
                           padding='post',
                           truncating='post') 

print([np.shape(i)for i in review_dummy])
   
#ONE HOT encoding for label
one_hot_encoder=OneHotEncoder(sparse=False)
sentiment_encoded=one_hot_encoder.fit_transform(np.expand_dims
                                                (sentiment_dummy,axis=-1))  


#train test split


X_train,X_test,y_train,y_test=train_test_split(review_dummy,sentiment_encoded,
                                               test_size=0.3,random_state=123)


X_train=np.expand_dims(X_train,axis=-1)
X_test=np.expand_dims(X_test, axis=-1)

print(one_hot_encoder.inverse_transform(np.expand_dims(y_train[0],axis=0)))

 

#%%Model Creation
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout,Embedding,Bidirectional



model=Sequential()
model.add(Embedding(num_words,64))
model.add(Bidirectional(LSTM(32,return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))
model.summary()

#%%Callbacks

from tensorflow.keras.callbacks import TensorBoard
import datetime

log_dir=os.path.join(PATH_LOGS,datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)


#%%COMPILE & MODEL TRAINING

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')

model.fit(X_train,y_train,epochs=1,
          validation_data=(X_test,y_test),
          callbacks=tensorboard_callback)

#%%model evaluation'
''''
predicted=[]

for test in X_test:
    predicted.append(model.predict(np.expand_dims(test, axis=0)))
'''
#preallocation of memory allocation

predicted_advanced=np.empty([len(X_test),2])
for index,test in enumerate(X_test):
    predicted_advanced[index,:]=model.predict(np.expand_dims(test, axis=0))
 
#%%model analysis
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
y_pred=np.argmax(predicted_advanced,axis=1)
y_true=np.argmax(y_test,axis=1)

print(classification_report(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))
print(accuracy_score(y_true,y_pred))

#%%model saving
model.save(MODEL_SAVE_PATH)
