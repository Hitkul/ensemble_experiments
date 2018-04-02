from mlens.ensemble import SuperLearner
from mlens.visualization import corrmat
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from pandas import DataFrame
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier


seed = 2017
np.random.seed(seed)

data = load_iris()
# idx = np.random.permutation(150)
X = data.data
y = data.target
print(X.shape)
print(y.shape)


from sklearn.preprocessing import LabelEncoder
encoder =  LabelEncoder()
y1 = encoder.fit_transform(y)
y = pd.get_dummies(y1).values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)





def model_1():
    model = Sequential()
    model.add(Dense(10,input_shape=(4,),activation='tanh'))
    model.add(Dense(8,activation='tanh'))
    model.add(Dense(6,activation='tanh'))
    model.add(Dense(3,activation='softmax'))
    model.compile(Adam(lr=0.04),'categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model

def model_2():
    model = Sequential()
    model.add(Dense(10,input_shape=(4,),activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(6,activation='relu'))
    model.add(Dense(3,activation='softmax'))
    model.compile(Adam(lr=0.04),'categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model


def model_3():
    model = Sequential()
    model.add(Dense(10,input_shape=(4,),activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(6,activation='relu'))
    model.add(Dense(3,activation='softmax'))
    model.compile(SGD(lr=0.04),'categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model




# def get_pred_of_model(m):
#     history = m.fit(X_train, y_train,epochs=100,verbose=0)
#     y_pred = m.predict(X_test)
#     y_test_class = np.argmax(y_test,axis=1)
#     y_pred_class = np.argmax(y_pred,axis=1)
#     print(accuracy_score(y_test_class,y_pred_class))
#     return y_pred_class

# pred_models = np.zeros((len(X_test),3))
# pred_models[:,0] = get_pred_of_model(model_1())
# pred_models[:,1] = get_pred_of_model(model_2())
# pred_models[:,2] = get_pred_of_model(model_3())

# pred_df = pd.DataFrame(pred_models)
# # print(pred_df.head())
# corrmat(pred_df.corr(), inflate=False)
# plt.show()

m1 = KerasClassifier(build_fn=model_1, epochs=100, verbose=0)
m2 = KerasClassifier(build_fn=model_2, epochs=100, verbose=0)
m3 = KerasClassifier(build_fn=model_3, epochs=100, verbose=0)

# --- Build ---
# Passing a scoring function will create cv scores during fitting
# the scorer should be a simple function accepting to vectors and returning a scalar
ensemble = SuperLearner(scorer=accuracy_score, random_state=seed, verbose=2)
# Build the first layer
ensemble.add([m1,m2,m3])
# Attach the final meta estimator
ensemble.add_meta(LogisticRegression())

# --- Use ---

# Fit ensemble
ensemble.fit(X_train, y_train)

# Predict
preds = ensemble.predict(X_test)

print("Fit data:\n%r" % ensemble.data)
print("Prediction score: %.3f" % accuracy_score(preds, y_test_class))