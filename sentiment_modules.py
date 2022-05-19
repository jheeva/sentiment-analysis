# -*- coding: utf-8 -*-
"""
Created on Thu May 19 03:58:39 2022

@author: End User
"""
#%%Module

import re
import os
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, Bidirectional
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json



class ExploratoryDataAnalysis():
    def _init_(self):
        pass
    
    
    '''to remove the HTML tags'''
    def remove_tags(self,data):
        for index,text in enumerate(data):
            data[index]=re.sub('<.*?>',text)
            
            return data
        
        
        '''to lower case  the data and split it'''
        def lower_split(self,data):
            for index,text in enumerate(data):
                data[index]=re.sub(['a-zA-Z'],' ',text).lower().split()
                
                return data
        
        def sentiment_tokenizer(self, data,token_save_path,num_words=10000, 
                                oov_token="<OOV>",prt=True):
                     # Build Token
            tokenizer = Tokenizer(num_words=num_words, oov_token=(oov_token))
            tokenizer.fit_on_texts(data)
                     
                     
            TOKENIZER_JSON_PATH=os.path.join(os.getcwd(), 'tokenizer_data.json')
            token_json=tokenizer.to_json()
                     
                     
                     # Save Token
            token_jason = tokenizer.to_json()
            with open(TOKENIZER_JSON_PATH , "w") as json_file:
             json.dump(token_jason, json_file)
             
             
             word_index = tokenizer.word_index 
                     
             if prt==False:
               print(word_index)
               print(dict(list(word_index.items())[0:10]))  
               
               
             data = tokenizer.texts_to_sequences(data)
                     
                     
                     
                     
    
 
    
 
    

#%%paths
PATH_LOGS=os.path.join(os.getcwd(),'logs')
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'model.h5')
url='https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'

df=pd.read_csv(url)
review=df['review']
review_dummy=review.copy()   #x_train

sentiment=df['sentiment']
sentiment_dummy=sentiment.copy()  #y_train


#%%
eda=ExploratoryDataAnalysis()
test=eda.remove_tags(review)
test=eda.lower_split(test)





#%% Model Creation

class ModelBuilding():
    
    def lstm_layer(self, num_words, nb_categories,
                   embedding_output=64, nodes=32, dropout=0.2):
        
        model = Sequential()
        model.add(Embedding(num_words, embedding_output)) # added the embedding layer, embedding doesn't need input, much faster and better with this approach
        model.add(Bidirectional(LSTM(nodes, return_sequences=True))) # added bidirectional
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes)))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation="softmax")) # softmax because classification between +ve & -ve
        model.summary()
        
        return model

    def simple_lstm_layer(self, num_words, nb_categories,
                   embedding_output=64, nodes=32, dropout=0.2):
        
        model = Sequential()
        model.add(Embedding(num_words, embedding_output)) 
        model.add(LSTM(nodes, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation="softmax")) 
        model.summary()
        
        return model
    
#%% Model evaluation

class ModelEvaluation():
    
    def evaluation(self, y_true, y_pred):
        print(classification_report(y_true, y_pred)) # classification report
        print(confusion_matrix(y_true, y_pred)) # confusion matrix
        print(accuracy_score(y_true, y_pred))

