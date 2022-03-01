import numpy as np
from sklearn.linear_model import LinearRegression
import preprocess as pp
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
#X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
#y = np.dot(X, np.array([1, 2])) + 3
#reg = LinearRegression().fit(X, y)
#reg.score(X, y)

#reg.coef_
#reg.intercept_
#reg.predict(np.array([[3, 5]]))


def toNumbers(data):
    clean = []
    for row in data:
        cleanPoint = []
        for point in row:
            cleanPoint.append(float(point))
        clean.append(cleanPoint)
    return clean


data = pp.readCSV("data/synthetic-2.csv")
data = toNumbers(data)
print(data)



model = LinearRegression().fit(data, [x[-1] for x in data])










fig, ax = plt.subplots()

xaxis = np.array([x[0] for x in data])
yaxis = np.array([x[1] for x in data])



x = [x[0] for x in data]
y = model.predict(data)

zipped = zip(x, y)
sort = sorted(zipped)
tuples = zip(*sort)
x, y = [ list(tuple) for tuple in  tuples]

X_Y_Spline = make_interp_spline(x, y)
y = X_Y_Spline(x)




ax.scatter(xaxis, yaxis)
plt.plot(x, y)

plt.show()