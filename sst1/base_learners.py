from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.optimizers import Adam


# set parameters:

batch_size = 32
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2



def cnn(vocab_size,n_dense,dropout,learning_rate,n_filters,filter_size,em,free_em_dim,number_of_classes):
    model = Sequential()
    if em == 'free':
        model.add(Embedding(vocab_size, free_em_dim,trainable = True))
    else:
        model.add(Embedding(Embedding(vocab_size, len(eval(em)[0]), weights = [eval(em)],input_length=length,trainable = True))
    model.add(Dropout(dropout))

    model.add(Conv1D(filters=n_filters, kernel_size=filter_size, activation='relu'))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(n_dense))
    model.add(Dropout(dropout))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(number_of_classes))
    model.add(Activation('sigmoid'))

    optimizer = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model