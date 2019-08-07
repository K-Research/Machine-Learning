import numpy
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from keras.models import Model
from keras.layers import Dense, Dropout, Input

cancer = load_breast_cancer()

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66)

def build_network(keep_prob = 0.5, optimizer = 'adam'):
    inputs = Input(shape = (30, ), name = 'input')
    x = Dense(512, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    x1 = Dense(256, activation = 'relu', name = 'hidden2')(x)
    x1 = Dropout(keep_prob)(x1)
    x2 = Dense(128, activation = 'relu', name = 'hidden3')(x1)
    x2 = Dropout(keep_prob)(x2)
    prediction = Dense(1, activation = 'sigmoid', name = 'output')(x2)
    model = Model(inputs = inputs, outputs = prediction)
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = numpy.linspace(0.1, 0.5, 5)
    return{"batch_size" : batches, "optimizer" : optimizers, "keep_prob" : dropout}

from keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn = build_network, verbose = 1)

hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

search = RandomizedSearchCV(model, hyperparameters, n_iter = 10, n_jobs = 1, cv = 3, verbose = 1)
search.fit(x_train, y_train)

print(search.best_params_)

score = search.score(x_test, y_test)
print("Score : ", score)

from sklearn.metrics import accuracy_score

y_pred = search.predict(x_test)