from sklearn.datasets import load_boston

boston = load_boston()
# print(boston.data.shape)
# print(boston.keys())
# print(boston.target)
# print(boston.target.shape)

x = boston.data
y = boston.target

# print(type(boston))

from sklearn.linear_model import LinearRegression # Lasso, Ridge

model= LinearRegression()