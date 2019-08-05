import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("‪‪../../Data/iris.csv", encoding = 'utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[ : , "Name"]
x = iris_data.loc[ : , ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

# y2 = iris_data.iloc[ : , 4]
# x2 = iris_data.iloc[ : , 0 : 4]

# print("==========================")
# print(x.shape)
# print(y.shape)

# print(x2.shape)
# print(y2.shape)

# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle = True)

