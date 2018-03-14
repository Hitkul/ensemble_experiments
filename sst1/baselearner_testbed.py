
# coding: utf-8

# In[21]:


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
from sklearn.metrics import accuracy_score
from base_learners import cnn,lstm,gru,bi_lstm,bi_gru,cnn_bi_gru,cnn_bi_lstm,cnn_gru,cnn_lstm


# In[2]:


def load_data_from_file(filename):
    with open(filename,'r', errors='ignore') as fin:
        lines = fin.readlines()
    label = [int(x.split()[0]) for x in lines]
    sentence = [' '.join(x.split()[1:]) for x in lines]
    return label,sentence


# In[3]:


train_labels,train_sentences = load_data_from_file('dataset/sst1/stsa.fine.train')
dev_label,dev_sentence = load_data_from_file('dataset/sst1/stsa.fine.dev')
test_labels,test_sentences = load_data_from_file('dataset/sst1/stsa.fine.test')


# In[4]:


train_sentences = train_sentences+dev_sentence
train_labels = train_labels+dev_label


# In[5]:


len(train_labels),len(train_sentences),len(test_labels),len(test_sentences)


# In[6]:


number_of_classes = len(set(train_labels))
number_of_classes


# In[7]:


def remove_punctuation(s):
    list_punctuation = list(string.punctuation)
    for i in list_punctuation:
        s = s.replace(i,'')
    return s


# In[8]:


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


# In[9]:


print("cleaning data")
trainX = [clean_sentence(s) for s in train_sentences]
testX = [clean_sentence(s) for s in test_sentences]
trainY = np.array(train_labels)


# In[10]:


back_up_for_fasttext = trainX


# In[11]:


max_len = 24


# In[12]:


def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# In[13]:


def encode_text(tokenizer, lines, length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded


# In[14]:


def load_GloVe_embedding(file_name):
    print('Loading GloVe word vectors.')
    embeddings_index = dict()
    f = open(file_name)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


# In[15]:


def get_GloVe_embedding_matrix(embeddings_index):
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# In[16]:


def load_fast_text_model(sentences):
    try:
        m = fasttext.load_model('fast_text_model.bin')
        print("trained model loaded")
        return m
    except:
        print("traning new model")
        with open('temp_file.txt','w') as temp_file:
            for sentence in sentences:
                temp_file.write(sentence)
        m = fasttext.cbow('temp_file.txt','fast_text_model')
        remove('temp_file.txt')
        print('model trained')
        return m


# In[17]:


def load_godin_word_embedding(path):
    print("Loading Goding model.")
    return godin_embedding.Word2Vec.load_word2vec_format(path, binary=True)


# In[18]:


def load_google_word2vec(file_name):
    print("Loading google news word2vec")
    return KeyedVectors.load_word2vec_format(file_name, binary=True)


# In[19]:


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


# In[20]:


tokenizer = create_tokenizer(trainX)
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % max_len)
print('Vocabulary size: %d' % vocab_size)
trainX = encode_text(tokenizer, trainX, max_len)
testX = encode_text(tokenizer, testX, max_len)
trainY = to_categorical(trainY,num_classes=number_of_classes)


# In[25]:


# glove_model = load_GloVe_embedding('word_embeddings/glove.6B.300d.txt')
# fast_text_model = load_fast_text_model(back_up_for_fasttext)
# godin_model = load_godin_word_embedding("../word_embeddings/word2vec_twitter_model.bin")
word2vec_model= load_google_word2vec('../word_embeddings/GoogleNews-vectors-negative300.bin')


# In[26]:


# embedding_matrix_glove = get_GloVe_embedding_matrix(glove_model)
embedding_matrix_word2vec = get_word_embedding_matrix(word2vec_model,300)
# embedding_matrix_fast_text = get_word_embedding_matrix(fast_text_model,100)
# embedding_matrix_godin = get_word_embedding_matrix(godin_model,400)


# In[22]:


pred_class_record = {}


# In[24]:


parameters_cnn = {
            "n_dense": 100,
            "dropout": 0.6,
            "learning_rate": 0.0001,
            "n_filters": 200,
            "filter_size": 3,
            "em": 'embedding_matrix_word2vec',
            "em_trainable_flag":True,
            "batch": 8,
            "epoch": 50
        }


# In[25]:


# parameters_lstm = {
#             "dropout": 0.5,
#             "learning_rate": 0.000641718,
#             "units_out": 64,
#             "em": 'embedding_matrix_word2vec',
#             "em_trainable_flag":False,
#             "batch": 8,
#             "epoch": 50
#         }


# parameters_gru = {
#             "dropout": 0.4,
#             "learning_rate": 0.000337372,
#             "units_out": 128,
#             "em": 'embedding_matrix_word2vec',
#             "em_trainable_flag":False,
#             "batch": 32,
#             "epoch": 50
#         }


# In[ ]:


model_cnn = cnn(length=max_len,
                vocab_size=vocab_size,
                n_dense=parameters_cnn['n_dense'],
                dropout=parameters_cnn['dropout'],
                learning_rate=parameters_cnn['learning_rate'],
                n_filters=parameters_cnn['n_filters'],
                filter_size=int(parameters_cnn['filter_size']),
                em = eval(parameters_cnn['em']),
                number_of_classes=number_of_classes,
                em_trainable_flag=parameters_cnn['em_trainable_flag'])


# In[ ]:


# model_lstm = lstm(length=max_len,
#              vocab_size=vocab_size,
#              learning_rate=parameters_lstm['learning_rate'],
#              dropout=parameters_lstm['dropout'],
#              units_out=parameters_lstm['units_out'],
#              em=parameters_lstm['em'],
#              number_of_classes=number_of_classes,
#              em_trainable_flag=parameters_lstm['em_trainable_flag'])


# model_gru = gru(length=max_len,
#              vocab_size=vocab_size,
#              learning_rate=parameters_gru['learning_rate'],
#              dropout=parameters_gru['dropout'],
#              units_out=parameters_gru['units_out'],
#              em=parameters_gru['em'],
#              number_of_classes=number_of_classes,
#              em_trainable_flag=parameters_gru['em_trainable_flag'])


# # In[ ]:


history_cnn = model_cnn.fit(trainX,trainY,epochs=parameters_cnn["epoch"],batch_size=parameters_cnn["batch"])


# # In[ ]:


# history_lstm = model_lstm.fit(trainX,trainY,epochs=parameters_lstm["epoch"],batch_size=parameters_lstm["batch"])


# history_gru = model_gru.fit(trainX,trainY,epochs=parameters_gru["epoch"],batch_size=parameters_gru["batch"])


# # In[ ]:


pred_cnn = model_cnn.predict(testX)
pred_class_cnn = [np.argmax(x) for x in pred_cnn]
acc_cnn = accuracy_score(test_labels,pred_class_cnn)


# In[ ]:


# pred_lstm = model_lstm.predict(testX)
# pred_class_lstm = [np.argmax(x) for x in pred_lstm]
# acc_lstm = accuracy_score(test_labels,pred_class_lstm)


# pred_gru = model_gru.predict(testX)
# pred_class_gru = [np.argmax(x) for x in pred_gru]
# acc_gru = accuracy_score(test_labels,pred_class_gru)


# # In[ ]:


# pred_class_record['cnn'] = {}
# pred_class_record['cnn']['pred_class'] = pred_class_cnn
# pred_class_record['cnn']['acc'] = acc_cnn
# pred_class_record['lstm'] = {}
# pred_class_record['lstm']['pred_class'] = pred_class_lstm
# pred_class_record['lstm']['acc'] = acc_lstm

# pred_class_record['gru'] = {}
# pred_class_record['gru']['pred_class'] = pred_class_gru
# pred_class_record['gru']['acc'] = acc_gru


# # In[ ]:


# with open('results/pred_results.json','w') as fout:
#     json.dump(pred_class_record,fout,indent=4)


# In[ ]:


print("cnn == ",acc_cnn)
# print("lstm == ",acc_lstm)
# print("gru == ",acc_gru)

