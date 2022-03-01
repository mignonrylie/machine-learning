#from turtle import update
import preprocess as pp
import matplotlib.pyplot as plt
import numpy as np

#loss/error function: just on a single data point (MSE: 1/M)
#cost function: over the entire data set (1/2M)



#hypothesis: t_0 + sum from i=1 to n of t_i*x_i
#where n = number of features, 11
#this would be a weight for each feature
#weight i times feature i of single data point x
def hypothesis(x, weights): #pass in a single data point, row of 11 features + label
    #assume bias is weights[0]
    sum = 0
    for index, weight in enumerate(weights):
        if index == 0:
            sum += weight
        else:
            #try:
            sum += weight * x[index-1]
            #except TypeError:
            #    sum += float(weight) * float(x[index-1])
    return sum


    sum = bias
    for index, weight in enumerate(weights):
        sum += weight * x[index]
    return sum

def rawError(x, weights): #pass in a single data point
    #return hypothesis(x, bias, weights) - x[-1] #assuming label is stored at last index
    #try:
    return hypothesis(x, weights) - x[-1] #assuming label is stored at last index
    #except TypeError:
    #    return hypothesis(x, weights) - float(x[-1])


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
            sum += error * point[theta-1]
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
#testData = [[2104, 460], [1416, 232], [1534, 315], [852, 178]]
#testWeights = [20, 0.25]
#a = 0.1

#print(testWeights)
#new = []
#for index, weight in enumerate(testWeights):
#    new.append(updateWeight(a, testData, testWeights, index))

#print(new)






data = pp.readCSV("data/winequality-red.csv")
titles = data[0]
del data[0]

def toNumbers(data):
    clean = []
    for row in data:
        cleanPoint = []
        for point in row:
            cleanPoint.append(float(point))
        clean.append(cleanPoint)
    return clean

data = toNumbers(data)

#print(titles)
#print(data)
weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] #t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11
newWeights = []
#alpha should be at most 0.0001
al = 0.0001

newWeights = updateAllWeights(al, data, weights)


while overallError(data, newWeights) > 100:
    newWeights = updateAllWeights(al, data, newWeights)

print(overallError(data, newWeights))
print(newWeights)

fig, ax = plt.subplots()
ax.scatter([x[0] for x in data], [x[1] for x in data])

#plt.show()




def graph(formula, x_range):  
    x = np.array(x_range)  
    y = formula(x)  # <- note now we're calling the function 'formula' with x
    plt.plot(x, y)  
    plt.show()  

def my_formula(x):


    return x**3+2*x-4
