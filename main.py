import datasets
X,Y = datasets.load_linear_example1()
print(X)
print(X[0])
print(Y)

import regression
model = regression.LinearRegression()
print(model.x)
model.fit(X,Y)
print(model.theta)
print(model.predict(X))
print(model.score(X,Y))

import datasets
X,Y = datasets.load_nonlinear_example1()
ex_X = datasets.polynomial2_features(X)
print(ex_X)
print(Y)

import datasets
X,Y = datasets.load_nonlinear_example1()
ex_X = datasets.polynomial3_features(X)
print(ex_X)
print(Y)
