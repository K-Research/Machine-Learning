import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *

# 기온 데이터 읽어 들이기
df = pd.read_csv("‪../../Data/tem10y.csv", encoding = "UTF-8")

# 데이터를 학습 전영과 테스트 전용으로 분리하기
train_year = (df["연"] <= 2015)
test_year = (df["연"] >= 2016)
interval = 6

# 과거 6일의 데이터를 기반으로 학습할 데이터 만들기
def make_data(data):
    x = [] # 학습 데이터
    y = [] # 결과
    temps = list(data["기온"])
    for i in range(len(temps)):
        if i < interval:
            continue
        y.append(temps[i])
        xa = []
        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)
    return(x, y)

train_x, train_y = make_data(df[train_year])
test_x, test_y = make_data(df[test_year])

train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

train_x = train_x.reshape((-1, 6, 1))
test_x = test_x.reshape((-1, 6, 1))

# 학습하기
model = Sequential()
model.add(Dense(128, input_shape = (6, 1), activation = 'relu'))
model.add(Dense(64))
model.add(TimeDistributed(Dense(32)))
model.add(LSTM(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

# 모델 컴파일
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])

# 모델 실행
model.fit(train_x, train_y, epochs = 100, batch_size = 1)

# 평가하기
pre_y = model.predict(test_x)
_, acc = model.evaluate(test_x, test_y, batch_size = 1)
print("acc : ", acc)

# 결과를 그래프로 그리기
plt.figure(figsize = (10, 6), dpi = 100)
plt.plot(test_y, c = 'r')
plt.plot(pre_y, c = 'b')
plt.savefig('tenki-kion-seqential.png')
plt.show()