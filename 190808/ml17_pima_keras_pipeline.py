from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
import numpy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# seed 값 생성
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = numpy.loadtxt("‪‪../../Data/pima-indians-diabetes.csv", delimiter = ",")

X = dataset[ : , 0 : 8]
Y = dataset[ : , 8]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 66, test_size = 0.2)

def build_network(keep_prob = 0.5, optimizer = 'adam'):
    inputs = Input(shape = (8, ), name = 'input')
    x = Sequential()(inputs)
    x1 = Dense(512, activation = 'relu', name = 'hidden1')(x)
    x2 = Dropout(keep_prob)(x1)
    x3 = Dense(256, activation = 'relu', name = 'hidden2')(x2)
    x4 = Dropout(keep_prob)(x3)
    x5 = Dense(128, activation = 'relu', name = 'hidden3')(x4)
    x6 = Dropout(keep_prob)(x5)
    prediction = Dense(1, activation = 'sigmoid', name = 'output')(x6)
    model = Model(inputs = inputs, outputs = prediction)
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = numpy.linspace(0.1, 0.5, 5)
    return{"model__batch_size" : batches, "model__optimizer" : optimizers, "model__keep_prob" : dropout}

model = KerasClassifier(build_fn = build_network, verbose = 1)

hyperparameters = create_hyperparameters()

pipe = Pipeline([("scaler", MinMaxScaler()), ('model', model)])

search = RandomizedSearchCV(pipe, hyperparameters, n_iter = 10, n_jobs = 1, cv = 3, verbose = 1)
search.fit(X_train, Y_train)

print(search.best_params_)

score = search.score(X_test, Y_test)
print("Score : ", score)