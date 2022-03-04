from turtle import update
import numpy as np
from sklearn.linear_model import LinearRegression
import preprocess as pp
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import univariate as model
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

def retrieveColumn(givenList, col):
    column = []
    for point in givenList:
        column.append(point[col])
    return column

#visualize
def visualize(data, weights):

    fig, ax = plt.subplots()

    xaxis = np.array([x[0] for x in data])
    yaxis = np.array([x[1] for x in data])



    x = [x[0] for x in data]
    y = model.createPoints(x, weights)

    zipped = zip(x, y)
    sort = sorted(zipped)
    tuples = zip(*sort)
    x, y = [ list(tuple) for tuple in  tuples]

    X_Y_Spline = make_interp_spline(x, y)
    y = X_Y_Spline(x)




    ax.scatter(xaxis, yaxis)
    plt.plot(x, y)

    plt.show()







def hypothesis(point, weights):
    sum = 0
    for n, weight in enumerate(weights):
        sum += weight * point[0]**n
    return sum

def rawError(point, weights):
    return hypothesis(point, weights) - point[-1]

def updateWeight(theta, data, alpha, weights):
    sum = 0
    for point in data:
        sum += rawError(point, weights) * point[0]**theta
    return weights[theta] - (alpha * (1/len(data)) * sum)

def batchUpdate(data, alpha, weights):
    newWeights = []
    for index, weight in enumerate(weights):
        newWeights.append(updateWeight(index, data, alpha, weights))
    return newWeights

def overallError(data, weights):
    sum = 0
    for point in data:
        sum += rawError(point, weights)**2
    return sum / len(data)

data = pp.readCSV("data/synthetic-2.csv")
data = toNumbers(data)
print(data)

data0 = [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]]
data1 = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]] #y = 1 + x
data2 = [[1, 2], [2, 6], [3, 12], [4, 20], [5, 30]] #y = x+ x^2

weight0 = [0, 0]
weight1 = [0, 0, 0]
weight2 = [0, 0, 0, 0]























def makeModel(data, weights, a, error):

    while (initError := overallError(data, weights)) > error:
        weights = batchUpdate(data, a, weights)
        if overallError(data, weights) > initError:
            a *= .25
        else:
            a *= 1.25
        #visualize(data, weights)
        FuncAnimation
        print(a, overallError(data, weights), weights)
    return weights

#makeModel(data0, weight0, 0.0001, 1)
#print(weight0)
#visualize(data2, weight0)

alpha = 0.001



numWeights = [2, 3, 5]

polyweights = []
for num in numWeights:
    newset = []
    for i in range(num):
        newset.append(0)
    polyweights.append(newset)

#newPolyWeights = []
#for weightSet in polyweights:
#    newWeights = []
#    #newWeights = updateAllWeights(a, data, weightSet)
#    newWeights = batchUpdate(data, alpha, weightSet)
#    while overallError(data, newWeights) > 10:
#    #for i in range(10):
#        #newWeights = updateAllWeights(a, data, weightSet)
#        newWeights = batchUpdate(data, alpha, weightSet)

 #   print(overallError(data, newWeights))
 #    newPolyWeights.append(newWeights)

newWeights = []
for weights in polyweights:
    newWeights.append(makeModel(data, weights, alpha, 10))

#print(hypothesis([0, 1], weight0))
#print(rawError([0, 1], weight0))
#print(updateWeight(0, data2, alpha, weight0))

news = [0, 0, 0, 0, 0, 0]

print("initial weights:")
#print(weight1)
print(news)
print(overallError(data, news))

news = makeModel(data, news, alpha, .4)

#while overallError(data, news) > .3:
 #   news = batchUpdate(data, alpha, news)
#weight1 = batchUpdate(data2, alpha, weight1)
print(news)
#print(weight1)
#weight1 = batchUpdate(data2, alpha, weight1)
#print(weight1)

#weight1 = makeModel(data2, weight1, alpha, .2)
#print(weight1)

visualize(data, newWeights[0])
