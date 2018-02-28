
import json
import string
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
import word2vecReader as godin_embedding
import fasttext
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input,Dense,Flatten,Dropout,Embedding
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.optimizers import Adam
from keras.layers.merge import concatenate
import keras.backend as K
from sklearn.model_selection import StratifiedKFold
from random import uniform,choice
from os import remove
import re
from sklearn.metrics import f1_score,precision_score,recall_score
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from pprint import pprint



def load_data_from_file(filename):
    data = pd.read_csv(filename, sep="\t", header=None)
    data.columns = ["tweet_id", "username", "database_id", "class","tweet"]
    return data





train_data = load_data_from_file('dataset/smm4h/personal_intake_tweets.txt')



dev_data = load_data_from_file('dataset/smm4h/personal_intake_tweets_dev.txt')



train_sentences = train_data['tweet'].tolist()+dev_data['tweet'].tolist()
train_labels = train_data['class'].tolist()+dev_data['class'].tolist()


def remove_punctuation(s):
    list_punctuation = list(punctuation)
    for i in list_punctuation:
        s = s.replace(i,'')
    return s


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


print("cleaning train data")
trainX = [clean_sentence(s) for s in train_sentences]
trainY = np.array([l-1 for l in train_labels])

max_len = 150


def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def encode_text(tokenizer, lines, length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded



#loading GloVe embedding
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





# create a weight matrix for words in training docs
def get_GloVe_embedding_matrix(embeddings_index):
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


#fast text word embedding
def load_fast_text_model(sentences):
    try:
        m = fasttext.load_model('word_embeddings/fast_text_model.bin')
        print("trained model loaded")
        return m
    except:
        print("traning new model")
        with open('temp_file.txt','w') as temp_file:
            for sentence in sentences:
                temp_file.write(sentence)
        m = fasttext.cbow('temp_file.txt','word_embeddings/fast_text_model')
        remove('temp_file.txt')
        print('model trained')
        return m



def get_fast_text_matrix(model):
    embedding_matrix = np.zeros((vocab_size,100))
    for word, i in tokenizer.word_index.items():
        try:
            embedding_vector = model[word]
        except KeyError:
            embedding_vector = None
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
    return embedding_matrix


#loading godin word embedding
def load_godin_word_embedding(path):
    print("Loading Goding model.")
    return godin_embedding.Word2Vec.load_word2vec_format(path, binary=True)


def get_godin_embedding_matrix(model):
    embedding_matrix = np.zeros((vocab_size,400))
    for word, i in tokenizer.word_index.items():
        try:
            embedding_vector = model[word]
        except KeyError:
            embedding_vector = None
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
    return embedding_matrix



#loading Google Word2Vec
def load_google_word2vec(file_name):
    print("Loading google news word2vec")
    return KeyedVectors.load_word2vec_format(file_name, binary=True)


def get_word2vec_embedding_matrix(model):
    embedding_matrix = np.zeros((vocab_size,300))
    for word, i in tokenizer.word_index.items():
        try:
            embedding_vector = model[word]
        except KeyError:
            embedding_vector = None
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
    return embedding_matrix



def define_model(length,vocab_size,n_dense,dropout,learning_rate,n_filters,filter_size_c1,filter_size_c2,filter_size_c3,em_c1,em_c2,em_c3,free_em_dim,em_trainable_flag_c1,em_trainable_flag_c2,em_trainable_flag_c3):
    # channel 1
    inputs1 = Input(shape=(length,))
    if em_c1 == 'free':
        embedding1 = Embedding(vocab_size, free_em_dim)(inputs1)
    else:
        embedding1 = Embedding(vocab_size, len(eval(em_c1)[0]), weights = [eval(em_c1)],input_length=length,trainable = em_trainable_flag_c1)(inputs1)
    
    conv1 = Conv1D(filters=n_filters, kernel_size=filter_size_c1, activation='relu')(embedding1)
    drop1 = Dropout(dropout)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # channel 2
    inputs2 = Input(shape=(length,))
    if em_c2 == 'free':
        embedding2 = Embedding(vocab_size, free_em_dim)(inputs2)
    else:
        embedding2 = Embedding(vocab_size, len(eval(em_c2)[0]), weights = [eval(em_c2)],input_length=length,trainable = em_trainable_flag_c2)(inputs2)
    conv2 = Conv1D(filters=n_filters, kernel_size=filter_size_c2, activation='relu')(embedding2)
    drop2 = Dropout(dropout)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    # channel 3
    inputs3 = Input(shape=(length,))
    if em_c3 == 'free':
        embedding3 = Embedding(vocab_size, free_em_dim)(inputs3)
    else:
        embedding3 = Embedding(vocab_size, len(eval(em_c3)[0]), weights = [eval(em_c3)],input_length=length,trainable = em_trainable_flag_c3)(inputs3)
    conv3 = Conv1D(filters=n_filters, kernel_size=filter_size_c3, activation='relu')(embedding3)
    drop3 = Dropout(dropout)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)
    # merge
    merged = concatenate([flat1, flat2, flat3])
    # interpretation
    dense1 = Dense(n_dense, activation='relu')(merged)
    outputs = Dense(3, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    # compile
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # summarize
#     print(model.summary())
    return model


tokenizer = create_tokenizer(trainX)
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % max_len)
print('Vocabulary size: %d' % vocab_size)
trainX = encode_text(tokenizer, trainX, max_len)


glove_model = load_GloVe_embedding('word_embeddings/glove.6B.300d.txt')
fast_text_model = load_fast_text_model(train_sentences)
godin_model = load_godin_word_embedding("word_embeddings/word2vec_twitter_model.bin")
word2vec_model= load_google_word2vec('word_embeddings/GoogleNews-vectors-negative300.bin')


embedding_matrix_glove = get_GloVe_embedding_matrix(glove_model)
embedding_matrix_word2vec = get_word2vec_embedding_matrix(word2vec_model)
embedding_matrix_fast_text = get_fast_text_matrix(fast_text_model)
embedding_matrix_godin = get_godin_embedding_matrix(godin_model)


seed = 7
np.random.seed(seed)


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


para_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform',name='learning_rate')

para_dropout = Real(low=0.4, high=0.9,name = 'dropout')

para_n_dense = Categorical(categories=[100,200,300,400], name='n_dense')

para_n_filters = Categorical(categories=[100,200,300,400],name='n_filters')

para_filter_size_c1 = Integer(low=1,high=6,name = 'filter_size_c1')
para_filter_size_c2 = Integer(low=1,high=6,name = 'filter_size_c2')
para_filter_size_c3 = Integer(low=1,high=6,name = 'filter_size_c3')

para_em_c1 = Categorical(categories=['embedding_matrix_fast_text','embedding_matrix_godin','embedding_matrix_word2vec','embedding_matrix_glove','free'],name='em_c1')
para_em_c2 = Categorical(categories=['embedding_matrix_fast_text','embedding_matrix_godin','embedding_matrix_word2vec','embedding_matrix_glove','free'],name='em_c2')
para_em_c3 = Categorical(categories=['embedding_matrix_fast_text','embedding_matrix_godin','embedding_matrix_word2vec','embedding_matrix_glove','free'],name='em_c3')

para_em_trainable_flag_c1 = Categorical(categories=[True,False],name='em_trainable_flag_c1')
para_em_trainable_flag_c2 = Categorical(categories=[True,False],name='em_trainable_flag_c2')
para_em_trainable_flag_c3 = Categorical(categories=[True,False],name='em_trainable_flag_c3')

para_free_em_dim = Categorical(categories=[100,300,400],name='free_em_dim')

para_batch_size = Categorical(categories=[8,16,32,64],name='batch_size')

# para_epoch = Categorical(categories=[10,20,30,50,100],name='epoch')
para_epoch = Categorical(categories=[1,2],name='epoch')


parameters = [para_learning_rate,para_dropout,para_n_dense,para_n_filters,para_filter_size_c1,para_filter_size_c2,para_filter_size_c3,para_em_c1,para_em_c2,para_em_c3,para_em_trainable_flag_c1,para_em_trainable_flag_c2,para_em_trainable_flag_c3,para_free_em_dim,para_batch_size,para_epoch]

default_parameters = [0.0071353667446707675,0.5777195655120914,400,100,6,4,4,'embedding_matrix_word2vec','embedding_matrix_word2vec','embedding_matrix_word2vec',False,True,False,100,16,1]

key = 1
record = {}



@use_named_args(dimensions=parameters)
def fitness(learning_rate,dropout,n_dense,n_filters,filter_size_c1,filter_size_c2,filter_size_c3,em_c1,em_c2,em_c3,em_trainable_flag_c1,em_trainable_flag_c2,em_trainable_flag_c3,free_em_dim,batch_size,epoch):
    global key
    global record
    print('-----------------------------combination no={0}------------------'.format(key))
    
    parameters = {
            "n_dense": n_dense,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "n_filters": n_filters,
            "filter_size_c1": filter_size_c1,
            "filter_size_c2": filter_size_c2,
            "filter_size_c3": filter_size_c3,
            "em_c1": em_c1,
            "em_c2": em_c2,
            "em_c3": em_c3,
            "free_em_dim": free_em_dim,
            "em_trainable_flag_c1": em_trainable_flag_c1,
            "em_trainable_flag_c2": em_trainable_flag_c2,
            "em_trainable_flag_c3": em_trainable_flag_c3,
            "batch": batch_size,
            "epoch": epoch
        }
    
    
    pprint(parameters)
    
    itr = 1
    f1_record = []
    p_record = []
    r_record = []
    itr_record = {}
    for train,test in kfold.split(trainX,trainY):
        print("k fold validation itr == {0}".format(itr))
        X = trainX[train]
        Y = to_categorical(trainY[train],num_classes=3)
        X_ = trainX[test]
        Y_ = list(trainY[test])
        model = define_model(length = max_len,
                             vocab_size=vocab_size,
                             n_dense = parameters["n_dense"],
                             dropout = parameters["dropout"],
                             learning_rate = parameters["learning_rate"],
                             n_filters = parameters["n_filters"],
                             filter_size_c1 = parameters["filter_size_c1"],
                             filter_size_c2 = parameters["filter_size_c2"],
                             filter_size_c3 = parameters["filter_size_c3"],
                             em_c1 = parameters["em_c1"],
                             em_c2 = parameters["em_c1"],
                             em_c3 = parameters["em_c1"],
                             free_em_dim = parameters["free_em_dim"],
                             em_trainable_flag_c1 = parameters["em_trainable_flag_c1"],
                             em_trainable_flag_c2 = parameters["em_trainable_flag_c2"],
                             em_trainable_flag_c3 = parameters["em_trainable_flag_c3"])
        history = model.fit([X,X,X],Y,epochs=parameters["epoch"],batch_size=parameters["batch"])
        pred = model.predict([X_,X_,X_])
        pred_labels = [x.argmax() for x in pred]

        f1 = f1_score(Y_,pred_labels,labels=[0,1],average='micro')
        p = precision_score(Y_,pred_labels,labels=[0,1],average='micro')
        r = recall_score(Y_,pred_labels,labels=[0,1],average='micro')
        print(f1,p,r)
        f1_record.append(f1)
        p_record.append(p)
        r_record.append(r)
        itr_record[itr] = {}
        itr_record[itr]["f1"] = f1
        itr_record[itr]["p"] = p
        itr_record[itr]["r"] = r
        model.save('models/'+str(key)+'_'+str(itr)+'.h5')
        itr+=1
    record[key] = {}
    record[key]["parameter"] = parameters
    mean_f1 = np.mean(f1_record)
    record[key]["mean_f1"] = mean_f1
    record[key]["itr_record"] = itr_record

    with open("models/record.json",'w')as fout:
        json.dump(record,fout,indent=4)
    key+=1
    
    del model
    K.clear_session()
    
    return -mean_f1




search_result = gp_minimize(func=fitness,
                            dimensions=parameters,
                            acq_func='EI',
                            n_calls=11,
                            x0=default_parameters)
