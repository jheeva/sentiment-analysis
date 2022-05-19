# -*- coding: utf-8 -*-
"""
Created on Thu May 19 00:04:51 2022

@author: End User
"""
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
from tensorflow.keras.models import load_model
import os
import re
from keras.preprocessing.sequence import pad_sequences

MODEL_PATH=os.path.join(os.getcwd(),'model.h5')
JSON_PATH=os.path.join(os.getcwd(),'tokenizer_data.json')


sentiment_classifier=load_model(MODEL_PATH)
sentiment_classifier.summary()

#Tokenizer Loading
with open (JSON_PATH,'r')as json_file:
    loaded_tokenizer=json.load(json_file)
    
 #%%deploy

new_review=['< br\> the review of new release movie']   

#data cleaning

for index,text in enumerate(list(new_review)):
    new_review[index]=re.sub('<.*?>',' ',text)
    
for index,text in enumerate(list(new_review)):
    new_review[index]=re.sub('[a-zA-Z]',' ',text).lower().split()
    
 #to vectorize the new review
loaded_tokenizer=tokenizer_from_json(loaded_tokenizer)   
new_review=pad_sequences(new_review,maxlen=200,truncating='post',padding='post')


#%%model prediction

import numpy as np

outcome=sentiment_classifier.predict(np.expand_dims(new_review, axis=-1))

print(np.argmax(outcome))

sentiment_dict={1:'postive',0:'negative'}

print('THIS REVIEW IS'+ sentiment_dict[np.argmax(outcome)] )
