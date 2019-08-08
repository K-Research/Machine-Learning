from keras.layers import Dense, Dropout, Input
from keras.models import Sequential, Model
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# 데이터 읽어 들이기
wine = pd.read_csv("‪../../Data/winequality-white.csv", sep = ";", encoding = "UTF-8")

# 데이터를 레이블과 데이터로 분리하기
y = wine["quality"]
x = wine.drop("quality", axis = 1)

'''
# y 레이블 변경하기
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist
'''

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

def build_network(keep_prob = 0.5, optimizer = 'adam'):
    inputs = Input(shape = (11, ), name = 'input')
    x = Sequential()(inputs)
    x1 = Dense(243, activation = 'relu', name = 'hidden1')(x)
    x2 = Dropout(keep_prob)(x1)
    x3 = Dense(81, activation = 'relu', name = 'hidden2')(x2)
    x4 = Dropout(keep_prob)(x3)
    x5 = Dense(27, activation = 'relu', name = 'hidden3')(x4)
    x6 = Dropout(keep_prob)(x5)
    x7 = Dense(9, activation = 'relu', name = 'hidden4')(x6)
    prediction = Dense(7, activation = 'softmax', name = 'output')(x7)
    model = Model(inputs = inputs, outputs = prediction)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    epochs = [10, 50, 100, 300, 500]
    return{"model__batch_size" : batches, "model__optimizer" : optimizers, "model__keep_prob" : dropout, "model__epochs" : epochs}

model = KerasClassifier(build_fn = build_network, verbose = 1)

hyperparameters = create_hyperparameters()

pipe = Pipeline([("scaler", MinMaxScaler()), ('model', model)])

search = RandomizedSearchCV(pipe, hyperparameters, n_iter = 10, n_jobs = 1, cv = 3, verbose = 1)

search.fit(x_train, y_train)

print(search.best_params_)