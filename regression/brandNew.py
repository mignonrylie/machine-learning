import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import preprocess as pp

absoluteValueError = False

def visualize(data, weights, title = None):
    fig, ax = plt.subplots()
    fig.suptitle(title)
    
    #print(data)
    xaxis = np.array([x[0] for x in data])
    yaxis = np.array([x[1] for x in data])

    x = np.arange(min(pp.retrieveColumn(data, 0)), max(pp.retrieveColumn(data, 0)), 0.1)
    y = [hypothesis([val], weights) for val in x]
    #hypothesis expects a list for point

    zipped = zip(x, y)
    sort = sorted(zipped)
    tuples = zip(*sort)
    x, y = [ list(tuple) for tuple in  tuples]

    X_Y_Spline = make_interp_spline(x, y)
    y = X_Y_Spline(x)

    ax.scatter(xaxis, yaxis)
    plt.plot(x, y)

    plt.show()



data1 = pp.toNumbers(pp.readCSV("data/synthetic-1.csv"))
data2 = pp.toNumbers(pp.readCSV("data/synthetic-2.csv"))

#print(data1)

#data1 = pp.standardize(data1)
#data2 = pp.standardize(data2)
#data1 = pp.squish(data1)
#data2 = pp.squish(data2)


#different parameters depending on the processing

#none
alpha = 1e-5 #0.001
errorBound = 1e-5
giveUp = 1e-100

#standardize
#alpha = 0.000001
#errorBound = 1e-5
#giveUp = 1e-300

#squish
#alpha = 1e-5
#errorBound = 1e-15
#giveUp = 1e-300

data = [data1, data2]

weights = []
numWeights = [2, 3, 5]
for num in numWeights:
    wSet = []
    for i in range(num + 1):
        wSet.append(0)
    weights.append(wSet)
    weights.append(wSet)

#multiply each part of the polynomial by x^n
def hypothesis(point, weights):
    sum = 0
    for index, weight in enumerate(weights):
        try:
            sum += weight * point[0]**index
        except (TypeError, IndexError) as e:
            print(e)
            #print(point, index)
            quit()
            #try:
            #    sum += weight * point**index
            #except:
            #    print(point, index)
    return sum

#simply the difference between the hypothesis and the actual value
def rawError(point, weights):
    if absoluteValueError:
        return abs(hypothesis(point, weights) - point[-1])
    return hypothesis(point, weights) - point[-1]

def updateWeights(data, weights, alpha):
    newWeights = []
    for index, weight in enumerate(weights):
        sum = 0
        
        #i am guessing that in basis expansion, the x that I multiply by at the very end is raised to that power.
        for point in data:
            try:
                sum += rawError(point, weights) * point[0]**index
            except (TypeError, IndexError):
                sum += rawError(point, weights) * point**index


        newWeights.append(weight - alpha * (1/len(data)) * sum)
    return newWeights

#calculates mean squared error across the entire data set
def overallError(data, weights):
    sum = 0
    for point in data:
        sum += rawError(point, weights)**2
    return sum / len(data)

#repeatedly updates the weights as long as the error is great enough
def generateModel(data, weights, alpha, error, delta):
    while((err := abs(overallError(data, weights))) > error):

        weights = updateWeights(data, weights, alpha)

        if abs((newErr := abs(overallError(data, weights))) - err) < delta: 
            return weights, newErr
        elif newErr > err:
            alpha *= .01
        else:
            alpha *= 1.5
    return weights, err

#alpha = 0.001
newWeights = []
errors = []
errorBound = 50
benchmarks = [35, .5, 10, .5, 10, .5]

#train each model
for index, weightSet in enumerate(weights):
    newWeightSet, error = generateModel(data[index % 2], weightSet, alpha, benchmarks[index], giveUp)
    newWeights.append(newWeightSet)
    errors.append(error)


titles = ["synthetic-1.csv", "synthetic-2.csv"]
for index, weightSet in enumerate(newWeights):
    print(weightSet)
    print(errors[index])
    visualize(data[index%2], weightSet, titles[index%2])
