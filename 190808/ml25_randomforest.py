import pandas as pd
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 데이터 읽어 들이기
wine = pd.read_csv("‪../../Data/winequality-white.csv", sep = ";", encoding = "UTF-8")

# 데이터를 레이블과 데이터로 분리하기
y = wine["quality"]
x = wine.drop("quality", axis = 1)

# y 레이블 변경하기
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

kfold_cv = KFold(n_splits = 5, shuffle = True)

parameters = {"max_depth" : [None, 1, 2, 3, 4, 5], "max_features" : ["auto", "sqrt", "log2"], "random_state" : [0], "class_weight" : [None, "balanced", "balanced_subsample"]}

# 학습하기
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv = kfold_cv)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)

# 평가하기
print("최적의 매개 변수 = ", model.best_estimator_)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답률 = ", accuracy_score(y_test, y_pred))
print(score)