from __future__ import print_function

import sys
sys.path.append('../')
import json
import string
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from os import remove
from pprint import pprint
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import keras.backend as K
from gensim.models import KeyedVectors
import word2vecReader as godin_embedding
import fasttext
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from base_learners import cnn,lstm,gru,bi_lstm,bi_gru,cnn_bi_gru,cnn_bi_lstm,cnn_gru,cnn_lstm


# In[8]:


def load_data_from_file(filename,test_flag = False):
    data = pd.read_csv(filename, sep="\t", header=None)
    if not test_flag:
        data.columns = ["tweet_id", "username", "database_id", "class","tweet"]
    else:
        data.columns = ["a", "b", "med","med", "tweet","class",]
    return data


# In[5]:


train_data = load_data_from_file('dataset/personal_intake_tweets.txt')
dev_data = load_data_from_file('dataset/personal_intake_tweets_dev.txt')


# In[6]:


train_sentences = train_data['tweet'].tolist()+dev_data['tweet'].tolist()
train_labels = train_data['class'].tolist()+dev_data['class'].tolist()


# In[9]:


test_data = load_data_from_file('dataset/task_2_test_full_form.txt',test_flag=True)


# In[14]:


test_labels = test_data['class'].tolist()
test_sentences = test_data['tweet'].tolist()


# In[21]:


train_labels = train_labels[:100]
train_sentences = train_sentences[:100]
test_labels = test_labels[:10]
test_sentences = test_sentences[:10]



# In[23]:


test_labels = [x-1 for x in test_labels]
train_labels = [x-1 for x in train_labels]


# In[24]:


number_of_classes = len(set(train_labels))



def remove_punctuation(s):
    list_punctuation = list(string.punctuation)
    for i in list_punctuation:
        s = s.replace(i,'')
    return s


# In[28]:


def clean_sentence(sentence):
    #removes links
    sentence = re.sub(r'(?P<url>https?://[^\s]+)', r'', sentence)
    # remove @usernames
    sentence = re.sub(r"\@(\w+)", "", sentence)
    #remove # from #tags
    sentence = sentence.replace('#','')
    # split into tokens by white space
    tokens = sentence.split()
    # remove punctuation from each token
    # should have used translate but for some reason it breaks on my server
    tokens = [remove_punctuation(w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    tokens = ' '.join(tokens)
    return tokens


# In[29]:


print("cleaning data")
trainX = [clean_sentence(s) for s in train_sentences]
testX = [clean_sentence(s) for s in test_sentences]
trainY = np.array(train_labels)


max_len = 20


# In[36]:


def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# In[37]:


def encode_text(tokenizer, lines, length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded


# In[38]:



def load_godin_word_embedding(path):
    print("Loading Goding model.")
    return godin_embedding.Word2Vec.load_word2vec_format(path, binary=True)


def get_word_embedding_matrix(model,dim):
    #dim = 300 for google word2vec
    #dim = 400 for godin
    #dim = 100 for fast text
    embedding_matrix = np.zeros((vocab_size,dim))
    for word, i in tokenizer.word_index.items():
        try:
            embedding_vector = model[word]
        except KeyError:
            embedding_vector = None
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
    return embedding_matrix


def get_results(model):
    pred = model_cnn.predict(testX)
    pred_class = [np.argmax(x) for x in pred]
    f1 = f1_score(test_labels,pred_class,labels=[0,1],average='micro')
    p = precision_score(test_labels,pred_class,labels=[0,1],average='micro')
    r = recall_score(test_labels,pred_class,labels=[0,1],average='micro')
    acc = accuracy_score(test_labels,pred_class)
    return f1,p,r,acc,pred_class

def add_record(m_name,f1_m,p_m,r_m,acc_m,pred_class_m):
    global pred_class_record
    pred_class_record[m_name]={}
    pred_class_record[m_name]['pred_class']=pred_class_m
    pred_class_record[m_name]['f1']=f1_m
    pred_class_record[m_name]['p']=p_m
    pred_class_record[m_name]['r']=r_m
    pred_class_record[m_name]['acc']=acc_m



# In[44]:


tokenizer = create_tokenizer(trainX)
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % max_len)
print('Vocabulary size: %d' % vocab_size)
trainX = encode_text(tokenizer, trainX, max_len)
testX = encode_text(tokenizer, testX, max_len)
trainY = to_categorical(trainY,num_classes=number_of_classes)


# In[25]:


godin_model = load_godin_word_embedding("../word_embeddings/word2vec_twitter_model.bin")



# In[26]:

embedding_matrix_godin = get_word_embedding_matrix(godin_model,400)



pred_class_record = {}


# In[24]:


parameters_cnn = {
            "n_dense": 100,
            "dropout": 0.7,
            "n_filters": 100,
            "filter_size": 2,
            "em": 'embedding_matrix_godin',
            "batch": 16,
            "epoch": 7
        }


# In[25]:


parameters_bi_lstm = {
            "dropout": 0.7,
            "units_out": 128,
            "em": 'embedding_matrix_godin',
            "batch": 8,
            "epoch": 9
        }

parameters_bi_lstm_cnn = {
            "n_filters":400,
            "filter_size":4,
            "em": 'embedding_matrix_godin',
            "conv_dropout":0.7,
            "l_or_g_dropout":0.2,
            "units_out":16,
            "batch": 8,
            "epoch": 12
        }

parameters_lstm_cnn = {
            "n_filters":300,
            "filter_size":2,
            "em": 'embedding_matrix_godin',
            "conv_dropout":0.8,
            "l_or_g_dropout":0.2,
            "units_out":128,
            "batch": 8,
            "epoch": 15
        }


# In[ ]:


model_cnn = cnn(length=max_len,
                vocab_size=vocab_size,
                n_dense=parameters_cnn['n_dense'],
                dropout=parameters_cnn['dropout'],
                n_filters=parameters_cnn['n_filters'],
                filter_size=int(parameters_cnn['filter_size']),
                em = eval(parameters_cnn['em']),
                number_of_classes=number_of_classes)

model_bi_lstm = bi_lstm(length=max_len,
                        vocab_size=vocab_size,
                        dropout=parameters_bi_lstm['dropout'],
                        units_out=parameters_bi_lstm['units_out'],
                        em=parameters_bi_lstm['em'],
                        number_of_classes=number_of_classes)

# In[ ]:


model_cnn_bi_lstm = cnn_bi_lstm(length=max_len,
                                vocab_size=vocab_size,
                                n_filters=parameters_bi_lstm_cnn['n_filters'],
                                filter_size=parameters_bi_lstm_cnn['filter_size'],
                                em=parameters_bi_lstm_cnn['em'],
                                number_of_classes=number_of_classes,
                                conv_dropout=parameters_bi_lstm_cnn['conv_dropout'],
                                l_or_g_dropout=parameters_bi_lstm_cnn['l_or_g_dropout'],
                                units_out=parameters_bi_lstm_cnn['units_out'])


model_cnn_lstm = cnn_lstm(length=max_len,
                                vocab_size=vocab_size,
                                n_filters=parameters_bi_lstm_cnn['n_filters'],
                                filter_size=parameters_bi_lstm_cnn['filter_size'],
                                em=parameters_bi_lstm_cnn['em'],
                                number_of_classes=number_of_classes,
                                conv_dropout=parameters_bi_lstm_cnn['conv_dropout'],
                                l_or_g_dropout=parameters_bi_lstm_cnn['l_or_g_dropout'],
                                units_out=parameters_bi_lstm_cnn['units_out'])



###CNN
history_cnn = model_cnn.fit(trainX,trainY,epochs=parameters_cnn["epoch"],batch_size=parameters_cnn["batch"])

f1_cnn,p_cnn,r_cnn,acc_cnn,pred_class_cnn = get_results(model_cnn)

add_record("cnn",f1_cnn,p_cnn,r_cnn,acc_cnn,pred_class_cnn)

model.save('models/cnn.h5')

##BI-LSTM
history_bi_lstm = model_bi_lstm.fit(trainX,trainY,epochs=parameters_bi_lstm["epoch"],batch_size=parameters_bi_lstm["batch"])

f1_bi_lstm,p_bi_lstm,r_bi_lstm,acc_bi_lstm,pred_class_bi_lstm = get_results(model_bi_lstm)

add_record("bi_lstm",f1_bi_lstm,p_bi_lstm,r_bi_lstm,acc_bi_lstm,pred_class_bi_lstm)

model.save('models/bi_lstm.h5')

##CNN_bi_lstm
history_cnn_bi_lstm = model_cnn_bi_lstm.fit(trainX,trainY,epochs=parameters_bi_lstm_cnn["epoch"],batch_size=parameters_bi_lstm_cnn["batch"])

f1_cnn_bi_lstm,p_cnn_bi_lstm,r_cnn_bi_lstm,acc_cnn_bi_lstm,pred_class_cnn_bi_lstm = get_results(model_cnn_bi_lstm)

add_record("cnn_bi_lstm",f1_cnn_bi_lstm,p_cnn_bi_lstm,r_cnn_bi_lstm,acc_cnn_bi_lstm,pred_class_cnn_bi_lstm)

model.save('models/cnn_bi_lstm.h5')

##CNN_lstm
history_cnn_lstm = model_cnn_lstm.fit(trainX,trainY,epochs=parameters_lstm_cnn["epoch"],batch_size=parameters_lstm_cnn["batch"])

f1_cnn_lstm,p_cnn_lstm,r_cnn_lstm,acc_cnn_lstm,pred_class_cnn_lstm = get_results(model_cnn_lstm)

add_record("cnn_lstm",f1_cnn_lstm,p_cnn_lstm,r_cnn_lstm,acc_cnn_lstm,pred_class_cnn_lstm)

model.save('models/cnn_lstm.h5')


with open("results/final_results.json",'w') as fout:
    json.dump(pred_class_record,fout,indent=4)


print("done")
