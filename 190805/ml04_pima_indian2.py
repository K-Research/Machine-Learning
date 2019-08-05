from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy
import tensorflow as tf

# seed 값 생성
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = numpy.loadtxt("‪‪../../Data/pima-indians-diabetes.csv", delimiter = ",")

X = dataset[ : , 0 : 8]
Y = dataset[ : , 8]

# 모델의 설정
model = KNeighborsClassifier(n_neighbors = 1)

# 모델 실행
model.fit(X, Y)

# 결과 출력
y_predict = model.predict(X)
print("\n Accuracy : %.4f" % accuracy_score(Y, y_predict))