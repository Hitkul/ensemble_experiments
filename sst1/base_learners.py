from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import LSTM,Bidirectional
from keras.optimizers import Adam




def cnn(length,vocab_size,learning_rate,n_dense,dropout,n_filters,filter_size,em,number_of_classes,em_trainable_flag):
    model = Sequential()
    model.add(Embedding(vocab_size, len(em[0]), weights = [em],input_length=length,trainable = em_trainable_flag))
    model.add(Dropout(dropout))
    model.add(Conv1D(filters=n_filters, kernel_size=filter_size, activation='relu'))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(n_dense))
    model.add(Dropout(dropout))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    if number_of_classes == 2:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
    else:
        model.add(Dense(number_of_classes))
        model.add(Activation('softmax'))

    optimizer = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model


def lstm(length,vocab_size,learning_rate,dropout,units_out,em,number_of_classes,em_trainable_flag):
    model = Sequential()
    model.add(Embedding(vocab_size, len(em[0]), weights = [em],input_length=length,trainable = em_trainable_flag))
    model.add(LSTM(units_out, dropout=dropout, recurrent_dropout=dropout))
    if number_of_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(number_of_classes, activation='softmax'))
    
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    print(model.summary())
    return model

def bi_lstm(length,vocab_size,learning_rate,dropout,units_out,em,number_of_classes,em_trainable_flag):
    model = Sequential()
    model.add(Embedding(vocab_size, len(em[0]), weights = [em],input_length=length,trainable = em_trainable_flag))
    model.add(Bidirectional(LSTM(units_out)))
    model.add(Dropout(dropout))
    if number_of_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(number_of_classes, activation='softmax'))
    
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    print(model.summary())
    return model


def cnn_bi_lstm(length,vocab_size,learning_rate,n_filters,filter_size,em,number_of_classes,em_trainable_flag,conv_dropout,l_or_g_dropout,units_out):
    model = Sequential()
    model.add(Embedding(vocab_size, len(em[0]), weights = [em],input_length=length,trainable = em_trainable_flag))
    model.add(Conv1D(filters=n_filters, kernel_size=filter_size, activation='relu'))
    model.add(Dropout(conv_dropout))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(units_out)))
    model.add(Dropout(l_or_g_dropout))
    if number_of_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(number_of_classes, activation='softmax'))
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    print(model.summary())
    return model


def cnn_lstm(length,vocab_size,learning_rate,n_filters,filter_size,em,number_of_classes,em_trainable_flag,conv_dropout,l_or_g_dropout,units_out):
    model = Sequential()
    model.add(Embedding(vocab_size, len(em[0]), weights = [em],input_length=length,trainable = em_trainable_flag))
    model.add(Conv1D(filters=n_filters, kernel_size=filter_size, activation='relu'))
    model.add(Dropout(conv_dropout))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units_out))
    model.add(Dropout(l_or_g_dropout))
    if number_of_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(number_of_classes, activation='softmax'))
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    print(model.summary())
    return model


