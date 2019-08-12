################### 데이터 ###################
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1 : ])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1 : ])))

################### 모델 구성 ###################
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model

# 인코딩될 표현(representation)의 크기
encoding_dim = 32

def build_network(keep_prob = 0.5, optimizer='adam'):
    # 입력 플레이스홀더
    input_img = Input(shape=(784, ))

    # # "encoded"는 입력의 인코딩된 표현
    # encoded = Dense(encoding_dim, activation='relu')(input_img)

    # encoded = Dense(64, activation='relu')(encoded) 
    # encoded = Dropout(dr)(encoded)
    # encoded = Dense(64, activation='relu')(encoded)
    # encoded = Dropout(dr)(encoded)
    # encoded = Dense(32, activation='relu')(encoded)
    # encoded = Dense(16, activation='relu')(encoded)
    # encoded = Dropout(dr)(encoded)
    # encoded = Dense(16, activation='relu')(encoded)

    # encoded = Dense(32, activation='relu')(encoded)
    # decoded = Dense(784, activation='sigmoid')(encoded)

    # 입력을 입력의 재구성으로 매핑할 모델
    autoencoder = Model(input_img, decoded)

    # 이 모델은 입력을 입력의 인코딩된 입력의 표현으로 매핑
    encoder = Model(input_img, encoded)

    # 인코딩된 입력을 위한 플레이스 홀더
    encoded_input = Input(shape = (encoding_dim, )) # 디코딩의 인풋레이어로 시작
    # 오토인코더 모델의 마지막 레이어 얻기
    decoder_layer = autoencoder.layers[-1]
    # 디코더 모델 생성
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder = Model(inputs=input_img, outputs=decoded)
    autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return autoencoder

def create_hyperparameters():
    batch_size = [1, 10, 20, 30, 40, 50]
    optimizer = ['adadelta', 'adam', 'rmsprop']
    dropout = np.linspace(0.1, 0.5, 5)
    epochs = [10, 50, 100, 300, 500]
    return {'batch_size' : batch_size, 'optimizer' : optimizer, 'keep_prob' : dropout, 'epochs' : epochs}

model = KerasClassifier(build_fn = build_network, verbose = 1)
hyperparameters = create_hyperparameters()
kfold_cv = KFold(n_splits = 5, shuffle = True)
                    
search = RandomizedSearchCV(estimator = model, param_distributions = hyperparameters, n_iter = 10, verbose = 1, cv = kfold_cv)

search.fit(x_train, x_train)

# loss, acc = search.evaluate(x_test, x_test)
# print(loss, acc)

print('Best parameter : ', search.best_params_)
print('Best estimator : ', search.best_estimator_)
print('Accuracy : ', search.score(x_test, x_test))

'''
# 숫자들을 인코딩 /디코딩
# test set에서 숫자들을 가져왔다는 것을 유의
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

print(encoded_imgs)
print(decoded_imgs)
print(encoded_imgs.shape) # (10000, 32)
print(decoded_imgs.shape) # (10000, 784)

################### 이미지 출력 ###################
# Matplotlib 사용
import matplotlib.pyplot as plt

n = 10 # 몇 개의 숫자를 나타낼 것인지
plt.figure(figsize = (20, 4))
for i in range(n):
    # 원본 데이터
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 재구성된 데이터
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

################### 그래프 출력 ###################
def plot_acc(history, title = None):
    # summmarize history for accuracy
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc = 0)
    # plt.show()

def plot_loss(history, title = None):
    # summmarize history for accuracy
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc = 0)
    # plt.show()

plot_acc(history, '(a) 학습 경과에 따른 정확도 변화 추이')
plt.show()
plot_loss(history, '(b) 학습 경과에 따른 손실값 변화 추이')
plt.show()

loss, acc = autoencoder.evaluate(x_test, x_test)
print(loss, acc)
'''