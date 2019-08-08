import pandas as pd
from sklearn.model_selection import train_test_split 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

iris_data = pd.read_csv("D:/Bitcamp/Data/iris.csv", encoding = "UTF-8", names = ['a', 'b', 'c', 'd', 'y'])

y = iris_data.loc[ : , "y"]
x = iris_data.loc[:, ["a", "b", "c", "d"]]

enc = OneHotEncoder()
y2 = enc.fit_transform(y.values.reshape(-1, 1)).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, y2, test_size = 0.2, train_size = 0.8, shuffle = True)

def build_network(keep_prob = 0.5, optimizer = 'adam'):
    inputs = Input(shape = (4, ), name = 'input')
    x = Sequential()(inputs)
    x1 = Dense(5, activation = 'relu', name = 'hidden1')(x)
    x2 = Dropout(keep_prob)(x1)
    x3 = Dense(8, activation = 'relu', name = 'hidden2')(x2)
    x4 = Dropout(keep_prob)(x3)
    x5 = Dense(10, activation = 'relu', name = 'hidden3')(x4)
    x6 = Dropout(keep_prob)(x5)
    prediction = Dense(3, activation = 'softmax', name = 'output')(x6)
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