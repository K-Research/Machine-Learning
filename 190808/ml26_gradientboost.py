import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

kfold_cv = KFold(n_splits = 5, shuffle = True)

parameters = {"loss" : ["ls", "lad", "huber", "quantile"], "learning_rate" : [0.1, 0.5, 1.0], "criterion" : ["mse", "friedman_mse"], "max_depth" : [None, 1, 2, 3, 4, 5], 
            "max_features" : [None, "auto", "sqrt", "log2"], "random_state" : [0]}

# 직선 회귀 분석하기
model = RandomizedSearchCV(GradientBoostingRegressor(), parameters, cv = kfold_cv)
model.fit(train_x, train_y) # 학습하기
pre_y = model.predict(test_x) # 예측하기

# RMSE 구하기
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(test_y, pre_y))

# R2 구하기
r2_y_predict = r2_score(test_y, pre_y)
print("R2 : ", r2_y_predict)