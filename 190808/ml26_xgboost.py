import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, RandomizedSearchCV
from xgboost import XGBRegressor
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

parameters = {"max_depth" : [1, 2, 3, 4, 5], "learning_rate" : [0.1, 0.5, 1.0], "n_estimators" : [100, 300, 500, 1000], "booster" : ["gbtree", "gblinear", "dart"], 
            "max_delta_step" : [1, 2, 3, 4, 5], "base_score" : [0.1, 0.2 ,0.3, 0.4, 0.5], "scale_pos_weight" : [0.1, 0.5, 1.0], "random_state" : [0], 
            "class_weight" : [None, "balanced", "balanced_subsample"], "missing" : [None, 0.1, 0.5, 1.0], 
            "importance_type" : ["gain", "weight", "cover", "total_gain", "total_cover"]}
            
# 직선 회귀 분석하기
model = RandomizedSearchCV(XGBRegressor(), parameters, cv = kfold_cv)
model.fit(train_x, train_y) # 학습하기
pre_y = model.predict(test_x) # 예측하기

# 평가하기
print("최적의 매개 변수 = ", model.best_estimator_)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답률 = ", accuracy_score(y_test, y_pred))
print(score)

# RMSE :  2.1875349013921324
# R2 :  0.9185750253196282