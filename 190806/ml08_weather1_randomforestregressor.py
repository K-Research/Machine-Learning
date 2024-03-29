import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

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

model = RandomForestRegressor(n_estimators = 100, max_features = "log2")
model.fit(train_x, train_y) # 학습하기
pre_y = model.predict(test_x) # 예측하기

# 결과를 그래프로 그리기
plt.figure(figsize = (10, 6), dpi = 100)
plt.plot(test_y, c = 'r')
plt.plot(pre_y, c = 'b')
plt.savefig('tenki-kion-lr.png')
plt.show()

score = model.score(test_x, test_y)
print(score)
print(classification_report(test_y, pre_y))
print("정답률 = ", accuracy_score(test_y, pre_y))