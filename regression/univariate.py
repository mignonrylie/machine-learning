import preprocess as pp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
#synthetic-1 and synthetic-2 are single-input datasets, i.e. univariate regression
#for univariate regression, h(x) = h_t(x) = t_0 + t_1*x
#choose parameters t such that the difference between h_t(x) and y is minimized
#the loss function (measure of distance) will be mean squared error:
#J(t_1) = 1/M * sum from i to M of (h_t(x^(i)) - y^(i))^2
#where x^(i), y^(i) is the i-th example, and M is the number of data points
#to update our parameters, we use gradient descent:
#t_j = t_j - a d/dt_j J(t)
#where t_j is the given parameter, a (0<a<1) is the learning rate, and d/dt_j is the partial derivative of J(t) w.r.t. the given parameter



#NEVERMIND LOL
#we'll be using polynomial regression on these datasets
#hypothesis = t_0 + sum i=1 to n of t_i * x^i
#for n = 2, 3, and 5

#give point and weights. number of weights is equal to n+1
def hypothesis(point, weights):
    sum = 0
    for n, weight in enumerate(weights):
        sum += weight * point[0]**n
    return sum

def rawError(point, weights):
    return hypothesis(point, weights) - point[-1]


#the measure of loss/error is mean squared error:
#J(t) = 1/M * sum from i to M of (h_t(x^(i)) - y^(i)) * x_parameter(theta)
def MSE(data, weights, theta): #pass in a data set-*, row of 11 features + label
    sum = 0
    for point in data:
        #error = rawError(point, bias, weights)
        error = rawError(point, weights)
        if theta == 0:
            sum += error
        else:
            sum += error * point[0]
        #sum += error^2
        #x = hypothesis(point, bias, weights)
        #difference = point[-1] - x
        #sum += difference^2
    return 1/len(data) * sum

#gradient descent to update weights:
#t_j = t_j - a d/dt_j J(t)*x_j (if j=0, x_j = 1. otherwise x_j = feature j of input example x)
#where t_j is the given parameter, a (0<a<1) is the learning rate, and d/dt_j is the partial derivative of J(t) w.r.t. the given parameter
def updateWeight(alpha, data, weights, j):
    theta = weights[j]
    #error = MSE(data, bias, weights, j)
    error = MSE(data, weights, j)
    return theta - alpha * error  


def overallError(data, weights):
    sum = 0
    for point in data:
        sum += rawError(point, weights)**2
    return sum / len(data)

def updateAllWeights(alpha, data, weights):
    newWeights = []
    for index, weight in enumerate(weights):
        newWeights.append(updateWeight(alpha, data, weights, index))
    return newWeights





def toNumbers(data):
    clean = []
    for row in data:
        cleanPoint = []
        for point in row:
            cleanPoint.append(float(point))
        clean.append(cleanPoint)
    return clean


data = pp.readCSV("data/synthetic-1.csv")
data = toNumbers(data)



numWeights = [2, 3, 5]

polyweights = []
for num in numWeights:
    newset = []
    for i in range(num+1):
        newset.append(0.1)
    polyweights.append(newset)

a = 0.00001

newPolyWeights = []
for weightSet in polyweights:
    newWeights = []
    newWeights = updateAllWeights(a, data, weightSet)
    while overallError(data, newWeights) > 1000:
    #for i in range(10):
        newWeights = updateAllWeights(a, data, weightSet)

    print(overallError(data, newWeights))
    newPolyWeights.append(newWeights)


poly0 = polyweights[0]
poly1 = polyweights[1]
poly2 = polyweights[2]


def calculateGuess(point, weights):
    ans = 0
    for n, weight in enumerate(weights):
        ans += weight * (point**n)
    return ans

def createPoints(datax, weights):
    ys = []
    for point in datax:
        ys.append(calculateGuess(point, weights))
    return ys


fig, ax = plt.subplots()

xaxis = np.array([x[0] for x in data])
yaxis = np.array([x[1] for x in data])



x = np.linspace(min(xaxis), max(xaxis), 500)
y = createPoints(x, poly2)

zipped = zip(x, y)
sort = sorted(zipped)
tuples = zip(*sort)
x, y = [ list(tuple) for tuple in  tuples]

X_Y_Spline = make_interp_spline(x, y)
y = X_Y_Spline(x)




ax.scatter(xaxis, yaxis)
plt.plot(x, y)

plt.show()

#graph(lambda x: poly0[0] + poly0[1]*x + poly0[2]*(x**2), np.arange(min(xaxis), max(xaxis)))
#graph(lambda x: poly1[0] + poly1[1]*x + poly1[2]*(x**2) + poly1[3]*(x**3), np.arange(min(xaxis), max(xaxis)))
#graph(lambda x: poly2[0] + poly2[1]*x + poly2[2]*(x**2) + poly2[3]*(x**3) + poly2[4]*(x**4) + poly2[5]*(x**5), np.arange(min(xaxis), max(xaxis)))