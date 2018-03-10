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
import codecs
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import keras.backend as K
from gensim.models import KeyedVectors
import word2vecReader as godin_embedding
import fasttext
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from base_learners import cnn,lstm,gru,bi_lstm,bi_gru,cnn_bi_gru,cnn_bi_lstm,cnn_gru,cnn_lstm


# In[2]:


def load_data_from_file(filename):
    with codecs.open(filename,'r', errors='ignore') as fin:
        lines = fin.readlines()
    label = [int(x.split()[0]) for x in lines]
    sentence = [' '.join(x.split()[1:]) for x in lines]
    return label,sentence


# In[3]:


train_labels,train_sentences = load_data_from_file('dataset/sst1/stsa.fine.train')
dev_label,dev_sentence = load_data_from_file('dataset/sst1/stsa.fine.dev')
test_labels,test_sentences = load_data_from_file('dataset/sst1/stsa.fine.test')


train_sentences = train_sentences+dev_sentence
train_labels = train_labels+dev_label


# In[6]:


number_of_classes = len(set(train_labels))



def remove_punctuation(s):
    list_punctuation = list(string.punctuation)
    for i in list_punctuation:
        s = s.replace(i,'')
    return s


# In[10]:
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


print("cleaning data")
trainX = [clean_sentence(s) for s in train_sentences]
testX = [clean_sentence(s) for s in test_sentences]
trainY = np.array(train_labels)


back_up_for_fasttext = trainX


# In[13]:


max_len = 24


# In[16]:


def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# In[17]:


def encode_text(tokenizer, lines, length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded


# In[18]:


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


# In[19]:


def get_GloVe_embedding_matrix(embeddings_index):
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# In[20]:


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


# In[21]:


def load_godin_word_embedding(path):
    print("Loading Goding model.")
    return godin_embedding.Word2Vec.load_word2vec_format(path, binary=True)


# In[22]:


def load_google_word2vec(file_name):
    print("Loading google news word2vec")
    return KeyedVectors.load_word2vec_format(file_name, binary=True)


# In[23]:


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


# In[24]:


tokenizer = create_tokenizer(trainX)
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % max_len)
print('Vocabulary size: %d' % vocab_size)
trainX = encode_text(tokenizer, trainX, max_len)
testX = encode_text(tokenizer, testX, max_len)
trainY = to_categorical(trainY,num_classes=number_of_classes)


# In[25]:


glove_model = load_GloVe_embedding('../word_embeddings/glove.6B.300d.txt')
fast_text_model = load_fast_text_model(back_up_for_fasttext)
godin_model = load_godin_word_embedding("../word_embeddings/word2vec_twitter_model.bin")
word2vec_model= load_google_word2vec('../word_embeddings/GoogleNews-vectors-negative300.bin')


# In[26]:


embedding_matrix_glove = get_GloVe_embedding_matrix(glove_model)
embedding_matrix_word2vec = get_word_embedding_matrix(word2vec_model,300)
embedding_matrix_fast_text = get_word_embedding_matrix(fast_text_model,100)
embedding_matrix_godin = get_word_embedding_matrix(godin_model,400)


# In[29]:


para_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform',name='learning_rate')
para_dropout = Categorical(categories=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],name = 'dropout')
para_em = Categorical(categories=['embedding_matrix_fast_text','embedding_matrix_godin','embedding_matrix_word2vec','embedding_matrix_glove'],name='em')
para_em_trainable_flag = Categorical(categories=[True,False],name='em_trainable_flag')
para_batch_size = Categorical(categories=[8,16,32,64],name='batch_size')
para_epoch = Categorical(categories=[5,10,20,50,100],name='epoch')

# para_units_out = Categorical(categories=[64,128,256,512], name='units_out')

# para_dropout_cnn_lstm = Categorical(categories=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],name = 'dropout')

para_n_dense = Categorical(categories=[100,200,300,400], name='n_dense')
para_n_filters = Categorical(categories=[32,100,200,300],name='n_filters')
para_filter_size = Integer(low=1,high=8,name = 'filter_size')


# In[30]:


parameters_cnn = [para_learning_rate,para_dropout,para_n_dense,para_n_filters,para_filter_size,para_em,para_em_trainable_flag,para_batch_size,para_epoch]
# parameters_lstm = [para_learning_rate,para_dropout,para_units_out,para_em,para_em_trainable_flag,para_batch_size,para_epoch]
# parameters_cnn_lstm = [para_learning_rate,para_dropout,para_dropout_cnn_lstm,para_units_out,para_n_filters,para_filter_size,para_em,para_em_trainable_flag,para_batch_size,para_epoch]


# In[33]:


default_parameters_cnn = [0.001,0.2,300,100,5,'embedding_matrix_word2vec',True,32,5]
# default_parameters_lstm = [0.001,0.2,128,embedding_matrix_word2vec,True,32,20]
# default_parameters_cnn_lstm = [0.001,0.2,0.2,128,100,5,embedding_matrix_word2vec,True,32,10]


# In[34]:


key = 1
record = {}


# In[38]:


##this will change based on the model
@use_named_args(dimensions=parameters_cnn)
def fitness(learning_rate,dropout,n_dense,n_filters,filter_size,em,em_trainable_flag,batch_size,epoch):
    global key
    global record
    global number_of_classes
    print('-----------------------------combination no={0}------------------'.format(key))
    parameters = {
            "learning_rate": learning_rate,    
            "dropout": dropout,
            "n_dense": n_dense,
            "n_filters": n_filters,
            "filter_size": filter_size,
            "em": em,
            "em_trainable_flag":em_trainable_flag,
            "batch": batch_size,
            "epoch": epoch
        }
    
    pprint(parameters)
    
    model = cnn(length=max_len,
                vocab_size=vocab_size,
                n_dense=parameters['n_dense'],
                dropout=parameters['dropout'],
                learning_rate=parameters['learning_rate'],
                n_filters=parameters['n_filters'],
                filter_size=int(parameters['filter_size']),
                em = eval(parameters['em']),
                number_of_classes=number_of_classes,
                em_trainable_flag=parameters['em_trainable_flag'])

    history = model.fit(trainX,trainY,epochs=parameters["epoch"],batch_size=parameters["batch"])
    pred = model.predict(testX)
    pred_class = [np.argmax(x) for x in pred]
    acc = accuracy_score(test_labels,pred_class)
    print(acc)
    record[key] = {}
    record[key]["parameter"] = parameters
    record[key]["acc"] = acc
    with open("results/cnn.json",'w')as fout:
        json.dump(record,fout,indent=4)
    key+=1
    
    del model
    K.clear_session()
    
    return -acc


# In[39]:


search_result = gp_minimize(func=fitness,
                            dimensions=parameters_cnn,
                            acq_func='EI',
                            n_calls=11,
                            x0=default_parameters_cnn)


# In[118]:


# parameters_cnn = {
#             "n_dense": 250,
#             "dropout": 0.2,
#             "learning_rate": 0.001,
#             "n_filters": 100,
#             "filter_size": 5,
#             "em": embedding_matrix_word2vec,
#             "em_trainable_flag":True,
#             "batch": 32,
#             "epoch": 5
#         }


# In[119]:


# parameters_lstm_or_gru = {
#             "dropout": 0.5,
#             "learning_rate": 0.001,
#             "units_out": 64,
#             "em": embedding_matrix_word2vec,
#             "em_trainable_flag":True,
#             "batch": 32,
#             "epoch": 4
#         }


# In[120]:


# parameters_cnn_lstm_or_gru = {
#             "n_filters": 100,
#             "filter_size": 5,
#             "conv_dropout": 0.5,
#             "l_or_g_dropout":0.5,
#             "learning_rate": 0.001,
#             "units_out": 64,
#             "em": embedding_matrix_word2vec,
#             "em_trainable_flag":True,
#             "batch": 32,
#             "epoch": 4
#         }


# In[121]:


# model = cnn(length=max_len,
#             vocab_size=vocab_size,
#             n_dense=parameters_cnn['n_dense'],
#             dropout=parameters_cnn['dropout'],
#             learning_rate=parameters_cnn['learning_rate'],
#             n_filters=parameters_cnn['n_filters'],
#             filter_size=parameters_cnn['filter_size'],
#             em = parameters_cnn['em'],
#             free_em_dim=parameters_cnn['free_em_dim'],
#             number_of_classes=number_of_classes,
#             em_trainable_flag=parameters_cnn['em_trainable_flag'])


# In[122]:


# model = lstm(length=max_len,
#              vocab_size=vocab_size,
#              learning_rate=parameters['learning_rate'],
#              dropout=parameters['dropout'],
#              units_out=parameters['units_out'],
#              em=parameters['em'],
#              number_of_classes=number_of_classes,
#              em_trainable_flag=parameters['em_trainable_flag'])


# In[123]:


# model = cnn_lstm(length=max_len,
#                     vocab_size=vocab_size,
#                     n_filters=parameters_cnn_lstm_or_gru['n_filters'],
#                     filter_size=parameters_cnn_lstm_or_gru['filter_size'],
#                     em=parameters_cnn_lstm_or_gru['em'],
#                     number_of_classes=number_of_classes,
#                     em_trainable_flag=parameters_cnn_lstm_or_gru['em_trainable_flag'],
#                     learning_rate=parameters_cnn_lstm_or_gru['learning_rate'],
#                     conv_dropout=parameters_cnn_lstm_or_gru['conv_dropout'],
#                     l_or_g_dropout=parameters_cnn_lstm_or_gru['l_or_g_dropout'],
#                     units_out=parameters_cnn_lstm_or_gru['units_out'])

