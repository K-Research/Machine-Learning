from keras import layers, models
from keras.applications import InceptionV3
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = numpy.dstack([x_train] * 48)
x_test = numpy.dstack([x_test] * 48)

x_train = x_train.reshape(x_train.shape[0], 112, 112, 3)
x_test = x_test.reshape(x_test.shape[0], 112, 112, 3)

# y_train = numpy.dstack([y_train] * 12)
# y_test = numpy.dstack([y_test] * 12)

# y_train = y_train.reshape(y_train.shape[0], 56, 56, 3)
# y_test = y_test.reshape(y_test.shape[0], 56, 56, 3)

# print(x_train.shape)
# print(x_test.shape)

# print(y_train.shape)
# print(y_test.shape)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train.shape)
# print(y_test.shape)

conv_base = InceptionV3(weights = 'imagenet', include_top = False, input_shape = (112, 112, 3))

# conv_base.summary()

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(10, activation = 'sigmoid'))

# model.summary()

model.compile(optimizer = 'adadelta', loss = 'binary_crossentropy', metrics = ['accuracy'])
# model.fit(x_train, y_train, epochs = 50, batch_size = 256, shuffle = True, validation_data = (x_test, y_test))
model.fit(x_train, y_train, epochs = 1, batch_size = 256, shuffle = True, validation_data = (x_test, y_test))

loss, acc = model.evaluate(x_test, y_test)
print(loss, acc)

# 0.010906125947041437 0.996289997291565