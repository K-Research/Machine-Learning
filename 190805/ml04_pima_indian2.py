from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy
import tensorflow as tf
from sklearn.model_selection import train_test_split

# seed 값 생성
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = numpy.loadtxt("‪‪../../Data/pima-indians-diabetes.csv", delimiter = ",")

X = dataset[ : , 0 : 8]
Y = dataset[ : , 8]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 66, test_size = 0.2)

# 모델의 설정
model = SVC()

# 모델 실행
model.fit(X_train, Y_train)

# 결과 출력
y_predict = model.predict(X_test)
print("\n Accuracy : %.4f" % accuracy_score(Y_test, y_predict))