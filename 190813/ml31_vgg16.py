from keras.applications import VGG16

conv_base = VGG16(weights = 'imagenet', include_top = False, input_shape = (150, 150, 3))
# conv_base = VGG16() # 224, 224, 3

conv_base.summary()

from keras import layers, models

model = models.Sequential()
model.add(conv_base)
# model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.summary()