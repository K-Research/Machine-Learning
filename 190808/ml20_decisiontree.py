from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state = 42)

tree = DecisionTreeClassifier(random_state = 0)
tree.fit(X_train, Y_train)
print("훈련 세트 정확도 : {:.3f}".format(tree.score(X_train, Y_train)))
print("테스트 세트 정확도 : {:.3f}".format(tree.score(X_test, Y_test)))

# tree = DecisionTreeClassifier(max_depth = 4, random_state = 0)
# tree.fit(X_train, Y_train)
# print("훈련 세트 정확도 : {:.3f}".format(tree.score(X_train, Y_train)))
# print("테스트 세트 정확도 : {:.3f}".format(tree.score(X_test, Y_test)))