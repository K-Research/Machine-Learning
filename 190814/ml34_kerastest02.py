from keras.callbacks import EarlyStopping
from keras.layers import Conv1D
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset_market = pandas.read_csv('C:/Document/Bitcamp/kospi2000test.csv', usecols = [1])
dataset_high = pandas.read_csv('C:/Document/Bitcamp/kospi2000test.csv', usecols = [2])
dataset_low = pandas.read_csv('C:/Document/Bitcamp/kospi2000test.csv', usecols = [3])
dataset_closing = pandas.read_csv('C:/Document/Bitcamp/kospi2000test.csv', usecols = [4])
dataset_volume = pandas.read_csv('C:/Document/Bitcamp/kospi2000test.csv', usecols = [5])
dataset_exchange = pandas.read_csv('C:/Document/Bitcamp/kospi2000test.csv', usecols = [6])

x_market = dataset_market
x_high = dataset_high
x_low = dataset_low
x_volume = dataset_volume
x_exchange = dataset_exchange

x_market = x_market.values.reshape(x_market.shape[0], x_market.shape[1])
x_high = x_high.values.reshape(x_high.shape[0], x_high.shape[1])
x_low = x_low.values.reshape(x_low.shape[0], x_low.shape[1])
x_volume = x_volume.values.reshape(x_volume.shape[0], x_volume.shape[1])
x_exchange = x_exchange.values.reshape(x_exchange.shape[0], x_exchange.shape[1])

scaler = MinMaxScaler()
scaler.fit(x_market)
x_market = scaler.transform(x_market)
x_high = scaler.transform(x_high)
x_low = scaler.transform(x_low)
x_volume = scaler.transform(x_volume)
x_exchange = scaler.transform(x_exchange)

x = numpy.concatenate((x_market, x_high, x_low, x_volume, x_exchange), axis = 1)
y = numpy.array(dataset_closing)

# print(dataset.shape)
# print(dataset[0])

size = 10

def split_10(seq, size):
    list = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i + size)]
        list.append(subset)
    return list

x = split_10(x, size)
x = numpy.array(x)

y = split_10(y, size)
y = numpy.array(y)

# print(dataset.shape)
# print(dataset[0])

# x = dataset[:, 0 : 10]
# y = dataset[:, 10 : ]

# print(x.shape)
# print(y.shape)

x = numpy.dstack([x] * 2)
y = numpy.dstack([y] * 10)

# print(x.shape)
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 66, test_size = 0.0001)

# print(x_train.shape)
# print(x_test.shape)

model = Sequential()

model.add(Conv1D(256, kernel_size = 1, input_shape = (10, 10), activation = 'relu'))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10))
model.add(Activation('linear'))

model.compile(loss = 'mse', optimizer = 'rmsprop', metrics = ['mse'])

early_stopping = EarlyStopping(monitor = 'mean_squared_error', patience = 100, mode = 'min')

model.fit(x_train, y_train, batch_size = 1, epochs = 100, validation_data = (x_test, y_test), callbacks = [early_stopping])

y_predict = model.predict(x_test)
print(y_predict)