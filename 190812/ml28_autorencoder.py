################### 데이터 ###################
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1 : ])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1 : ])))

print(x_train.shape) # (60000, 784)
print(x_test.shape) # (10000, 784)

################### 모델 구성 ###################
from keras.layers import Dense, Input
from keras.models import Model

# 인코딩될 표현(representation)의 크기
encoding_dim = 32

# 입력 플레이스홀더
input_img = Input(shape = (784, ))

# "encoded"는 입력의 인코딩된 표현
encoded = Dense(encoding_dim, activation = 'relu')(input_img)

# "decoded"는 입력의 손실있는 재구성 (lossy reconstruction)
decoded = Dense(784, activation = 'sigmoid')(encoded)
# decoded = Dense(784, activation = 'relu)(encoded)

# 입력을 입력의 재구성으로 매핑할 모델
autoencoder = Model(input_img, decoded) # 784 -> 32 -> 784

# 이 모델은 입력을 입력의 인코딩된 입력의 표현으로 매핑
encoder = Model(input_img, encoded) # 784 -> 32