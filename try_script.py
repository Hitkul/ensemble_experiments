from mlens.ensemble import SuperLearner
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from pandas import DataFrame
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam


seed = 2017
np.random.seed(seed)

data = load_iris()
idx = np.random.permutation(150)
X = data.data[idx]
y = data.target[idx]
print(X.shape)
print(y.shape)



from sklearn.preprocessing import LabelEncoder
encoder =  LabelEncoder()
y1 = encoder.fit_transform(y)

y = pd.get_dummies(y1).values



def model_1():
    model = Sequential()
    model.add(Dense(10,input_shape=(4,),activation='tanh'))
    model.add(Dense(8,activation='tanh'))
    model.add(Dense(6,activation='tanh'))
    model.add(Dense(3,activation='softmax'))
    model.compile(Adam(lr=0.04),'categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model

m = model_1()
history = m.fit(X[:75], y[:75],epochs=100)

y_pred = m.predict(X[75:])
print(y_pred)

# # --- Build ---
# # Passing a scoring function will create cv scores during fitting
# # the scorer should be a simple function accepting to vectors and returning a scalar
# ensemble = SuperLearner(scorer=accuracy_score, random_state=seed, verbose=2)
# # Build the first layer
# ensemble.add([RandomForestClassifier(random_state=seed), SVC()])
# # Attach the final meta estimator
# ensemble.add_meta(LogisticRegression())

# # --- Use ---

# # Fit ensemble
# ensemble.fit(X[:75], y[:75])

# # Predict
# preds = ensemble.predict(X[75:])

# print("Fit data:\n%r" % ensemble.data)
# print("Prediction score: %.3f" % accuracy_score(preds, y[75:]))