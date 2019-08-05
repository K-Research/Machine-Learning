from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x_data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_data = np.array([0, 0, 0, 1])

# 2. 모델
model = Sequential()
model.add(Dense(64, input_dim = 2, activation = 'sigmoid'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

# 3. 실행
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_data, y_data, epochs = 2000)

# 4. 평가 예측
x_test = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_predict = np.round(model.predict(x_test))

# 4. 평가 예측
print(y_predict, "의 예측 결과 : ", y_predict)
_, acc = model.evaluate(x_test, y_data, batch_size = 1)
print("acc : ", acc)