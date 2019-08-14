from keras import layers, models
from keras.applications import VGG16
from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# x_train = numpy.dstack([x_train] * 12)
# x_test = numpy.dstack([x_test] * 12)

# x_train = x_train.reshape(x_train.shape[0], 56, 56, 3)
# x_test = x_test.reshape(x_test.shape[0], 56, 56, 3)

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

# print(x_train.shape)
# print(x_test.shape)

# conv_base = VGG16(weights = 'imagenet', include_top = False, input_shape = (150, 150, 3))
conv_base = VGG16(weights = 'imagenet', include_top = False, input_shape = (32, 32, 3))
# conv_base = VGG16() # 224, 224, 3

conv_base.summary()

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))

model.summary()

model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 2, batch_size = 256, shuffle = True, validation_data = (x_test, y_test))

loss, acc = model.evaluate(x_test, y_test)
print(loss, acc)