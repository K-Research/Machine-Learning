from sklearn.datasets import load_boston

boston = load_boston()
# print(boston.data.shape)
# print(boston.keys())
# print(boston.target)
# print(boston.target.shape)

x = boston.data
y = boston.target

# print(type(boston))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.02)

from sklearn.linear_model import LinearRegression, Lasso, Ridge

model = LinearRegression()
# model = Lasso()
# model = Ridge()

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(score)