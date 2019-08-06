import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# 데이터 읽어 들이기
wine = pd.read_csv("‪../../Data/winequality-white.csv", sep = ";", encoding = "UTF-8")

# 데이터를 레이블과 데이터로 분리하기
y = wine["quality"]

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(y.values.reshape(-1, 1)).toarray()

x = wine.drop("quality", axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 66, test_size = 0.2)

# 학습하기
model = Sequential()
model.add(Dense(49, input_dim = 11, activation = 'relu'))
model.add(Dense(7, activation = 'softmax'))

# 모델 컴파일
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# 모델 실행
model.fit(x_train, y_train, epochs = 100, batch_size = 1)

# 평가하기
y_predict = model.predict(x_test)
_, acc = model.evaluate(x_test, y_test, batch_size = 1)
print("acc : ", acc)