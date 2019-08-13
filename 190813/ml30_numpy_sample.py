import numpy
import pandas

from keras.datasets import boston_housing

(X_train_boston_housing_keras, y_train_boston_housing_keras), (X_test_boston_housing_keras, y_test_boston_housing_keras) = boston_housing.load_data()
numpy.save("boston_housing_keras.npy", [X_train_boston_housing_keras, y_train_boston_housing_keras, X_test_boston_housing_keras, y_test_boston_housing_keras])

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
numpy.save("breast_cancer.npy", cancer)

from sklearn.datasets import load_boston

boston = load_boston()
numpy.save("boston_sklearn.npy", boston)

from keras.datasets import cifar10

(X_train_cifar10, y_train_cifar10), (X_test_cifar10, y_test_cifar10) = cifar10.load_data()
numpy.save("cifar10.npy", [X_train_cifar10, y_train_cifar10, X_test_cifar10, y_test_cifar10])

iris_data = pandas.read_csv("C:/Document/Bitcamp/Data/iris.csv", encoding = "UTF-8")
numpy.save("iris.npy", iris_data)

from keras.datasets import mnist

(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()
numpy.save("mnist.npy", [X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist])

datasets = numpy.loadtxt("C:/Document/Bitcamp/Data/pima-indians-diabetes.csv", delimiter = ",")
numpy.save("pima-indians-diabetes.npy", datasets)

wine = pandas.read_csv("C:/Document/Bitcamp/Data/winequality-white.csv", sep = ",", encoding = "UTF-8")
numpy.save("winequality-white.npy", wine)