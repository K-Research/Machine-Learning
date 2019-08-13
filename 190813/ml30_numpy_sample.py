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
# numpy.save("cifar10.npy", [X_train_cifar10, y_train_cifar10, X_test_cifar10, y_test_cifar10])
cifar10_x = numpy.vstack((numpy.array(X_train_cifar10), numpy.array(X_test_cifar10)))
cifar10_y = numpy.vstack((numpy.array(y_train_cifar10), numpy.array(y_test_cifar10)))
numpy.save("cifar10_x.npy", cifar10_x)
numpy.save("cifar10_y.npy", cifar10_y)

iris_data = pandas.read_csv("C:/Document/Bitcamp/Data/iris.csv", encoding = "UTF-8")
numpy.save("iris.npy", iris_data)

iris2_data = pandas.read_csv("C:/Document/Bitcamp/Data/iris2.csv", encoding = "UTF-8")
numpy.save("iris2.npy", iris_data)

from keras.datasets import mnist

(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()
# numpy.save("mnist.npy", [X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist])
mnist_x = numpy.vstack((numpy.array(X_train_mnist), numpy.array(X_test_mnist)))
mnist_y = numpy.hstack((numpy.array(y_train_mnist), numpy.array(y_test_mnist)))
numpy.save("mnist_x.npy", mnist_x)
numpy.save("mnist_y.npy", mnist_y)

datasets = numpy.loadtxt("C:/Document/Bitcamp/Data/pima-indians-diabetes.csv", delimiter = ",")
numpy.save("pima-indians-diabetes.npy", datasets)

tem10y_data = pandas.read_csv("C:/Document/Bitcamp/Data/tem10y.csv", encoding = "UTF-8")
numpy.save("tem10y.npy", tem10y_data)

wine = pandas.read_csv("C:/Document/Bitcamp/Data/winequality-white.csv", sep = ",", encoding = "UTF-8")
numpy.save("winequality-white.npy", wine)