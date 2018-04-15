from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
# load data
dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model on training data
model = XGBClassifier()


# learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
# param_grid = dict(learning_rate=learning_rate)
param_grid={
    'max_depth': [3,4,5,6,7,8,9],
    'subsample': [0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    'colsample_bytree': [0.5,0.6,0.7,0.8],
    'n_estimators': [1000,2000,3000]
    'reg_alpha': [0.01, 0.02, 0.03, 0.04]
}

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold,verbose=1)
grid_result = grid_search.fit(X, Y)


best_est = grid_result.best_estimator_


# make predictions for test data
y_pred = best_est.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
