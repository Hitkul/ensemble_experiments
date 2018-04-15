
# coding: utf-8

# In[1]:

from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from base_learners import cnn,lstm,bi_lstm,cnn_bi_lstm,cnn_lstm
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [12,10]
from mlens.visualization import corrmat
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import codecs

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


# In[4]:


train_sentences = train_sentences+dev_sentence
train_labels = train_labels+dev_label


# In[5]:


len(train_labels),len(train_sentences),len(test_labels),len(test_sentences)


# In[6]:


train_labels = train_labels[:500]
train_sentences = train_sentences[:500]
test_labels=test_labels[:100]
test_sentences = test_sentences[:100]


# In[7]:


number_of_classes = len(set(train_labels))
# number_of_classes


# In[8]:


print(len(train_labels),len(train_sentences),len(test_labels),len(test_sentences))


# In[9]:


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


# In[11]:


print("cleaning data")
trainX = [clean_sentence(s) for s in train_sentences]
testX = [clean_sentence(s) for s in test_sentences]
trainY = np.array(train_labels)
testY=test_labels


# In[12]:


max_len = 24


# In[13]:


def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# In[14]:


def encode_text(tokenizer, lines, length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded


# In[15]:


def load_godin_word_embedding(path):
    print("Loading Goding model.")
    return godin_embedding.Word2Vec.load_word2vec_format(path, binary=True)


# In[16]:


def load_google_word2vec(file_name):
    print("Loading google news word2vec")
    return KeyedVectors.load_word2vec_format(file_name, binary=True)


# In[17]:


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


# In[18]:


tokenizer = create_tokenizer(trainX)
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % max_len)
print('Vocabulary size: %d' % vocab_size)
trainX = encode_text(tokenizer, trainX, max_len)
testX = encode_text(tokenizer, testX, max_len)
trainY = to_categorical(trainY,num_classes=number_of_classes)


# In[19]:


# godin_model = load_godin_word_embedding("../word_embeddings/word2vec_twitter_model.bin")
word2vec_model= load_google_word2vec('../word_embeddings/GoogleNews-vectors-negative300.bin')


# In[20]:


embedding_matrix_word2vec = get_word_embedding_matrix(word2vec_model,300)
# embedding_matrix_godin = get_word_embedding_matrix(godin_model,400)


# ## base models

# In[21]:


cnn_parameter = {'batch': 8,
                'dropout': 0.6,
                'em': 'embedding_matrix_word2vec',
                'em_trainable_flag': True,
                'epoch': 10,
                'filter_size': 6,
                'learning_rate': 0.0001,
                'n_dense': 200,
                'n_filters': 100}

lstm_parameter={'batch': 64,
                    'dropout': 0.6,
                    'em': 'embedding_matrix_word2vec',
                    'em_trainable_flag': False,
                    'epoch': 20,
                    'learning_rate': 0.0034157107277860235,
                    'units_out': 128}

cnn_lstm_parameter={'batch': 8,
                    'conv_dropout': 0.5,
                    'em': 'embedding_matrix_word2vec',
                    'em_trainable_flag': False,
                    'epoch': 10,
                    'filter_size': 1,
                    'learning_rate': 0.001,
                    'lstm_dropout': 0.4,
                    'n_filters': 100,
                    'units_out': 64}

cnn_bi_lstm_parameter={'batch': 8,
                    'conv_dropout': 0.5,
                    'em': 'embedding_matrix_word2vec',
                    'em_trainable_flag': False,
                    'epoch': 5,
                    'filter_size': 1,
                    'learning_rate': 0.001,
                    'lstm_dropout': 0.2,
                    'n_filters': 100,
                    'units_out': 64}

bi_lstm_parameter={'batch':8,
                 'dropout': 0.6,
                 'em': 'embedding_matrix_word2vec',
                 'em_trainable_flag': False,
                 'epoch': 5,
                 'learning_rate': 0.0001,
                 'units_out': 256}


# In[22]:


# cnn
# 0.4710
def init_cnn():
    return cnn(length=max_len,
               vocab_size=vocab_size,
               learning_rate=cnn_parameter['learning_rate'],
               n_dense=cnn_parameter['n_dense'],
               dropout=cnn_parameter['dropout'],
               n_filters=cnn_parameter['n_filters'],
               filter_size=cnn_parameter['filter_size'],
               em=eval(cnn_parameter['em']),
               number_of_classes=number_of_classes,
               em_trainable_flag=cnn_parameter['em_trainable_flag'])


# In[23]:


#  lstm
# 0.4701
def init_lstm():
    return lstm(length=max_len,
                vocab_size=vocab_size,
                learning_rate=lstm_parameter['learning_rate'],
                dropout=lstm_parameter['dropout'],
                units_out=lstm_parameter['units_out'],
                em=eval(lstm_parameter['em']),
                number_of_classes=number_of_classes,
                em_trainable_flag=lstm_parameter['em_trainable_flag'])


# In[24]:


# bi_lstm
# 0.4529
def init_bi_lstm():
    return bi_lstm(length=max_len,
                vocab_size=vocab_size,
                learning_rate=bi_lstm_parameter['learning_rate'],
                dropout=bi_lstm_parameter['dropout'],
                units_out=bi_lstm_parameter['units_out'],
                em=eval(bi_lstm_parameter['em']),
                number_of_classes=number_of_classes,
                em_trainable_flag=bi_lstm_parameter['em_trainable_flag'])


# In[25]:


# cnn_lstm
# 0.4179
def init_cnn_lstm():
    return cnn_lstm(length=max_len,
                    vocab_size=vocab_size,
                    learning_rate=cnn_lstm_parameter['learning_rate'],
                    n_filters=cnn_lstm_parameter['n_filters'],
                    filter_size=cnn_lstm_parameter['filter_size'],
                    em=eval(cnn_lstm_parameter['em']),
                    number_of_classes=number_of_classes,
                    em_trainable_flag=cnn_lstm_parameter['em_trainable_flag'],
                    conv_dropout=cnn_lstm_parameter['conv_dropout'],
                    l_or_g_dropout=cnn_lstm_parameter['lstm_dropout'],
                    units_out=cnn_lstm_parameter['units_out'])


# In[26]:


# cnn_bi_lstm
# 0.4705
def init_cnn_bi_lstm():
    return cnn_bi_lstm(length=max_len,
                    vocab_size=vocab_size,
                    learning_rate=cnn_bi_lstm_parameter['learning_rate'],
                    n_filters=cnn_bi_lstm_parameter['n_filters'],
                    filter_size=cnn_bi_lstm_parameter['filter_size'],
                    em=eval(cnn_bi_lstm_parameter['em']),
                    number_of_classes=number_of_classes,
                    em_trainable_flag=cnn_bi_lstm_parameter['em_trainable_flag'],
                    conv_dropout=cnn_bi_lstm_parameter['conv_dropout'],
                    l_or_g_dropout=cnn_bi_lstm_parameter['lstm_dropout'],
                    units_out=cnn_bi_lstm_parameter['units_out'])


# In[27]:


def get_pred_of_model(m,epoch,batch,trainX,trainY,testX,testY):
    history = m.fit(trainX,trainY,epochs=epoch,batch_size=batch,verbose=2)
    pred = m.predict(testX)    
    pred_class = np.argmax(pred,axis=1)
    pred_class=pred_class.astype(int)
    acc = accuracy_score(testY,pred_class)
    print(acc)
#     print(pred)
    return acc,pred_class,pred


# In[28]:


pred_prob_base = np.zeros((len(testX),number_of_classes,5))
pred_class_base = np.zeros((len(testX),5),dtype=np.int32)
acc_results={}


# In[29]:


cnn_base = init_cnn()
acc_results['cnn'],pred_class_base[:,0],pred_prob_base[:,:,0] = get_pred_of_model(cnn_base,cnn_parameter['epoch'],cnn_parameter['batch'],trainX,trainY,testX,testY)


# In[30]:


lstm_base = init_lstm()
acc_results['lstm'],pred_class_base[:,1],pred_prob_base[:,:,1] = get_pred_of_model(lstm_base,lstm_parameter['epoch'],lstm_parameter['batch'],trainX,trainY,testX,testY)


# In[31]:


bi_lstm_base=init_bi_lstm()
acc_results['bi_lstm'],pred_class_base[:,2],pred_prob_base[:,:,2] = get_pred_of_model(bi_lstm_base,bi_lstm_parameter['epoch'],bi_lstm_parameter['batch'],trainX,trainY,testX,testY)


# In[32]:


cnn_lstm_base = init_cnn_lstm()
acc_results['cnn_lstm'],pred_class_base[:,3],pred_prob_base[:,:,3] = get_pred_of_model(cnn_lstm_base,cnn_lstm_parameter['epoch'],cnn_lstm_parameter['batch'],trainX,trainY,testX,testY)


# In[33]:


cnn_bi_lstm_base = init_cnn_bi_lstm()
acc_results['cnn_bi_lstm'],pred_class_base[:,4],pred_prob_base[:,:,4] = get_pred_of_model(cnn_bi_lstm_base,cnn_bi_lstm_parameter['epoch'],cnn_bi_lstm_parameter['batch'],trainX,trainY,testX,testY)


# In[34]:


print(acc_results)


# In[35]:


# pred_class_base[:10]


# In[36]:


# pred_prob_base[:2]


# ## Analyzing performance of base models 

# In[37]:


number_of_base_models = 5


# In[38]:


correct_predicted_by_all = 0
incorrect_predicted_by_all = 0
correct_predicted_by_some=[0 for _ in range(number_of_base_models-1)] #index0 = correct predicted by 1, index1 = correct predicted by 2 and so on.


# In[39]:


# pred_class_base[0],np.bincount(pred_class_base[0]),len(np.bincount(pred_class_base[0])),testY[0]


# In[40]:


for x,y in zip(pred_class_base,testY):
    bin_count = np.bincount(x)
    if len(bin_count)<=y or bin_count[y]==0:
        incorrect_predicted_by_all+=1
    elif bin_count[y] == number_of_base_models:
        correct_predicted_by_all+=1
    else:
        correct_predicted_by_some[bin_count[y]-1]+=1


# In[41]:


print(incorrect_predicted_by_all,correct_predicted_by_all,correct_predicted_by_some)


# In[42]:


if sum(correct_predicted_by_some)+correct_predicted_by_all+incorrect_predicted_by_all == len(testY):
    print("results look good")
else:
    print("something went wrong")


# In[43]:


acc_results['base_model_counts']={}


# In[44]:


acc_results['base_model_counts']['correct_predicted_by_all'] = correct_predicted_by_all
acc_results['base_model_counts']['incorrect_predicted_by_all'] = incorrect_predicted_by_all
acc_results['base_model_counts']['correct_predicted_by_some'] = correct_predicted_by_some


# In[45]:


print(acc_results)


# ## prediction corelation

# In[46]:


pred_df = pd.DataFrame(pred_class_base)
pred_df.columns = ["cnn","lstm","bi_lstm","cnn_lstm","cnn_bi_lstm"]


# In[47]:


# pred_df.head()


# In[48]:


corrmat(pred_df.corr(), inflate=False,show=False)
plt.savefig('results/corr_matrix_base_xg.png', bbox_inches='tight')
# corrmat(pred_df.corr(), inflate=False)


# ## average

# In[49]:


avg_pred_prob = pred_prob_base.mean(axis=2)


# In[50]:


avg_pred_class = np.argmax(avg_pred_prob,axis=1)
avg_pred_class=avg_pred_class.astype(int)


# In[51]:


acc = accuracy_score(testY,avg_pred_class)
# acc


# In[52]:


acc_results['average'] = acc


# In[53]:


# acc_results


# In[54]:


pred_df['average']=avg_pred_class


# ## Majority

# In[55]:


majority_pred_class = [int(np.argmax(np.bincount(x))) for x in pred_class_base]


# In[56]:


acc = accuracy_score(testY,majority_pred_class)
# acc


# In[57]:


acc_results['majority'] = acc


# In[58]:


# acc_results


# In[59]:


pred_df['majority']=majority_pred_class


# ## Blend ensemble

# In[60]:


seed=42


# In[61]:


baseX, devX, baseY, devY = train_test_split(trainX, train_labels, test_size=0.10, random_state=seed)


# In[62]:


baseY = np.array(baseY)
baseY = to_categorical(baseY,num_classes=number_of_classes)


# In[63]:


print(len(baseX),len(baseY),len(devX),len(devY))


# In[64]:


metaX = np.zeros((len(devY),5),dtype=np.int32)


# In[65]:


_,metaX[:,0],_ = get_pred_of_model(init_cnn(),cnn_parameter['epoch'],cnn_parameter['batch'],baseX,baseY,devX,devY)


# In[66]:


_,metaX[:,1],_ = get_pred_of_model(init_lstm(),lstm_parameter['epoch'],lstm_parameter['batch'],baseX,baseY,devX,devY)


# In[67]:


_,metaX[:,2],_ = get_pred_of_model(init_bi_lstm(),bi_lstm_parameter['epoch'],bi_lstm_parameter['batch'],baseX,baseY,devX,devY)


# In[68]:


_,metaX[:,3],_ = get_pred_of_model(init_cnn_lstm(),cnn_lstm_parameter['epoch'],cnn_lstm_parameter['batch'],baseX,baseY,devX,devY)


# In[69]:


_,metaX[:,4],_ = get_pred_of_model(init_cnn_bi_lstm(),cnn_bi_lstm_parameter['epoch'],cnn_bi_lstm_parameter['batch'],baseX,baseY,devX,devY)


# In[70]:


print(len(devY),len(metaX))


# Logistic Regressing meta model

# In[71]:


meta_model=LogisticRegression()


# In[72]:


meta_model.fit(metaX,devY)


# In[73]:


blend_pred_class = meta_model.predict(pred_class_base)


# In[74]:


acc = accuracy_score(testY,blend_pred_class)
# acc


# In[75]:


acc_results['blend'] = acc


# In[76]:


print(acc_results)


# In[77]:


pred_df['blend']=blend_pred_class


# XGboost meta model

# In[78]:


meta_model_xg = XGBClassifier()


# In[79]:


# param_grid={
#     'max_depth': [3,4,5,6,7,8,9],
#     'subsample': [0.4,0.5,0.6,0.7,0.8,0.9,1.0],
#     'colsample_bytree': [0.5,0.6,0.7,0.8],
#     'n_estimators': [1000,2000,3000],
#     'reg_alpha': [0.01, 0.02, 0.03, 0.04]
# }

param_grid={
    'max_depth': [3,4],
    'subsample': [0.6,0.7],
    'colsample_bytree': [0.6,0.7],
    'n_estimators': [1000,2000],
    'reg_alpha': [0.01, 0.02]
}


# In[80]:


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)


# In[81]:


grid_search = GridSearchCV(meta_model_xg, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold,verbose=1)


# In[82]:


grid_result = grid_search.fit(metaX,devY)


# In[ ]:


best_est_blen_xg = grid_result.best_estimator_


# In[167]:


blend_pred_class_xg = best_est_blen_xg.predict(pred_class_base)


# In[171]:


acc = accuracy_score(testY,blend_pred_class_xg)
acc


# In[172]:


acc_results['blend_xg'] = acc


# In[173]:


print(acc_results)


# In[174]:


pred_df['blend_xg']=blend_pred_class_xg


# ## Stacked ensemble

# In[175]:


np.random.seed(seed)


# In[176]:


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)


# In[177]:


stacked_metaX=np.array([[0, 0, 0, 0,0]],dtype=np.int64)


# In[178]:


stacked_metaY = []


# In[179]:


count=1


# In[180]:


trainX=[clean_sentence(x) for x in train_sentences]


# In[181]:


trainY=train_labels


# In[182]:


trainY=np.array(trainY)
trainX=np.array(trainX)


# In[183]:


for train,test in kfold.split(trainX,trainY):
    print("----------------------itr = {}--------------".format(count))
    stacked_trainX = list(trainX[train])
    stacked_trainY = list(trainY[train])
    stacked_testX = list(trainX[test])
    stacked_testY = list(trainY[test])
    
    tokenizer = create_tokenizer(stacked_trainX)
    vocab_size = len(tokenizer.word_index) + 1
    stacked_trainX = encode_text(tokenizer, stacked_trainX, max_len)
    stacked_testX = encode_text(tokenizer, stacked_testX, max_len)
    stacked_trainY = to_categorical(stacked_trainY,num_classes=number_of_classes)
    
    embedding_matrix_word2vec = get_word_embedding_matrix(word2vec_model,300)
    
    for i in stacked_testY:
        stacked_metaY.append(i)
        
    temp = np.zeros((len(stacked_testY),5),dtype=np.int64)
    
    _,temp[:,0],_ = get_pred_of_model(init_cnn(),cnn_parameter['epoch'],cnn_parameter['batch'],stacked_trainX,stacked_trainY,stacked_testX,stacked_testY)
    _,temp[:,1],_ = get_pred_of_model(init_lstm(),lstm_parameter['epoch'],lstm_parameter['batch'],stacked_trainX,stacked_trainY,stacked_testX,stacked_testY)
    _,temp[:,2],_ = get_pred_of_model(init_bi_lstm(),bi_lstm_parameter['epoch'],bi_lstm_parameter['batch'],stacked_trainX,stacked_trainY,stacked_testX,stacked_testY)
    _,temp[:,3],_ = get_pred_of_model(init_cnn_lstm(),cnn_lstm_parameter['epoch'],cnn_lstm_parameter['batch'],stacked_trainX,stacked_trainY,stacked_testX,stacked_testY)
    _,temp[:,4],_ = get_pred_of_model(init_cnn_bi_lstm(),cnn_bi_lstm_parameter['epoch'],cnn_bi_lstm_parameter['batch'],stacked_trainX,stacked_trainY,stacked_testX,stacked_testY)
    
    stacked_metaX = np.concatenate((stacked_metaX, temp), axis=0)
    count+=1


# In[184]:


# len(stacked_metaX),len(stacked_metaY)


# In[185]:


stacked_metaX = np.delete(stacked_metaX, (0), axis=0)


# In[186]:


#temp line
# stacked_metaY=stacked_metaY[-500:]


# In[187]:


# len(stacked_metaX),len(stacked_metaY)


# In[188]:


stacked_meta_model=LogisticRegression()


# In[189]:


stacked_meta_model.fit(stacked_metaX,stacked_metaY)


# In[190]:


stacked_pred_class = stacked_meta_model.predict(pred_class_base)


# In[191]:


acc = accuracy_score(testY,stacked_pred_class)
# acc


# In[192]:


acc_results['stacked'] = acc


# In[193]:


print(acc_results)


# In[194]:


pred_df['stacked']=stacked_pred_class


# xgboost meta model

# In[83]:


# stacked_metaX
# stacked_metaY


# In[ ]:


stacked_meta_model_xg = XGBClassifier()


# In[ ]:


grid_search = GridSearchCV(stacked_meta_model_xg, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold,verbose=1)


# In[ ]:


grid_result = grid_search.fit(stacked_metaX,stacked_metaY)


# In[ ]:


best_est_stacked_xg = grid_result.best_estimator_


# In[ ]:


stacked_pred_class_xg = best_est_stacked_xg.predict(pred_class_base)


# In[ ]:


acc = accuracy_score(testY,stacked_pred_class_xg)
# acc


# In[204]:


acc_results['stacked_xg'] = acc


# In[205]:


print(acc_results)


# In[206]:


pred_df['stacked_xg']=stacked_pred_class


# ## Prediction Correlation

# In[207]:


corrmat(pred_df.corr(), inflate=False,show=False)
plt.savefig('results/corr_matrix_full_xg.png', bbox_inches='tight')
# corrmat(pred_df.corr(), inflate=False)


# ## saving results

# In[208]:


with open('results/ens_result_xg.json','w') as fout:
    json.dump(acc_results,fout,indent=4)

