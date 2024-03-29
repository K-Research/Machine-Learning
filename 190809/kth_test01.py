import h5py
from keras.backend import tensorflow_backend
from keras.datasets import cifar10
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow
import warnings

# config = tensorflow.ConfigProto()
# config.gpu_options.allow_growth = True
# tensorflow_backend.set_session(tensorflow.Session(config = config))

# warnings.filterwarnings('ignore')
# warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.simplefilter(action = 'ignore', category = UserWarning)
# warnings.resetwarnings()

# CIFAR_10은 3채널로 구성된 32x32 이미지 60000장을 갖는다.
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

# 상수 정의
BATCH_SIZE = 128
NB_EPOCH = 20
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

# 데이터셋 불러오기
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# print('X_train shape : ', X_train.shape)
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')

X_train = X_train[ : 300]
X_test = X_test[ : 300]
y_train = y_train[ : 300]
y_test = y_test[ : 300]

# 범주형으로 전환
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# 실수형으로 지정하고 정규화
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_train = X_train.flatten()
X_test = X_test.flatten()
X_train = X_train.reshape(X_train.shape[0], 1)
X_test = X_test.reshape(X_test.shape[0], 1)

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(X_train)
scaler.fit(X_test)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.flatten()
X_test = X_test.flatten()

X_train = X_train.reshape(-1, 3)
X_test = X_test.reshape(-1, 3)

X_train = X_train.reshape(-1, 32, 3)
X_test = X_test.reshape(-1, 32, 3)

X_train = X_train.reshape(-1, 32, 32, 3)
X_test = X_test.reshape(-1, 32, 32, 3)

# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)

def build_network(keep_prob = 0.5, optimizer = 'adam'):
    inputs = Input(shape = (IMG_ROWS, IMG_COLS, IMG_CHANNELS), name = 'input')
    x = Sequential()(inputs)
    x1 = Conv2D(32, (3, 3), padding = 'same', name = 'hidden1')(x)
    x2 = Activation('relu')(x1)
    x3 = MaxPooling2D(pool_size = (2, 2))(x2)
    x4 = Dropout(keep_prob)(x3)
    x5 = Flatten()(x4)
    x6 = Dense(512, name = 'hidden2')(x5)
    x7 = Activation('relu')(x6)
    x8 = Dense(512, name = 'hidden3')(x7)
    x9 = Activation('relu')(x8)
    x10 = Dropout(0.5)(x9)
    x11 = Dense(NB_CLASSES, name = 'hidden4')(x10)
    prediction = Activation('softmax', name = 'output')(x11)
    model = Model(inputs = inputs, outputs = prediction)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def create_hyperparameters():
    batches = [1, 10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    epochs = [10, 50, 100, 300, 500]
    return{"batch_size" : batches, "optimizer" : optimizers, "keep_prob" : dropout, "epochs" : epochs}

data_generator = ImageDataGenerator(featurewise_center = True, featurewise_std_normalization = True, rotation_range = 180, width_shift_range = 1.0, height_shift_range = 1.0, 
                                    horizontal_flip = True, vertical_flip = True, fill_mode = 'nearest')

for i in range(len(X_train)):
    img = X_train[i, :, :, :]
    img = img.reshape((1,)+ img.shape)  
    i = 0

    for batch in data_generator.flow(img, batch_size= 1):
      i += 1
      X_train = np.vstack((X_train, batch))
      Y_train = np.vstack((Y_train, Y_train[i]))
#       print(X_train.shape)
#       print(Y_train.shape)
          
    #  if i == 199:
      if i == 32:
        break

# print("X_train shape: ", X_train.shape)
# print("X_test shape: ", X_test.shape)
# print("Y_train shape: ", Y_train.shape)
# print("Y_test shape: ", Y_test.shape)

model = KerasClassifier(build_fn = build_network, verbose = 1)

hyperparameters = create_hyperparameters()

# search = RandomizedSearchCV(model, hyperparameters, n_iter = 10, n_jobs = -1, cv = 3, verbose = 1)
search = RandomizedSearchCV(model, hyperparameters, n_iter = 10, cv = 3, verbose = 1)
search.fit(X_train, Y_train)

print(search.best_params_)

score = search.score(X_test, Y_test)
print("Score : ", score)

'''
# 학습
import keras

tb_hist = keras.callbacks.TensorBoard(log_dir = './graph', histogram_freq = 0, write_graph = True, write_images = True)

from keras.callbacks import ModelCheckpoint, EarlyStopping

early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10)

history = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = NB_EPOCH, validation_split = VALIDATION_SPLIT, verbose = VERBOSE, callbacks = [early_stopping_callback, tb_hist])

print('Testing...')
score = model.evaluate(X_test, Y_test, batch_size = BATCH_SIZE, verbose = VERBOSE)
print("\nTest scroe : ", score[0])
print('Test accuracy : ', score[1])

# 모델 저장
# model.json = model.to_json()
# open('cifar10_architecture.json', 'w').write(model_json)
# model.save_weights('cifar10_weights.h5', overwrite = True)

# 히스토리에 있는 모든 데이터 나열
print(history.history.keys())

# 단순 정확도에 대한 히스토리 요악
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

# 손실에 대한 히스토리 요약
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()
'''

# {'optimizer': 'adam', 'keep_prob': 0.1, 'epochs': 100, 'batch_size': 30}
# 300/300 [==============================] - 1s 3ms/step
# Score :  0.08000000230967999