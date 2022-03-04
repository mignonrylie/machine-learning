import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import preprocess as pp

from sklearn.metrics import mean_squared_error

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
#weights length [2, 2, 3, 3, 5, 5]

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

def rawError(point, weights):
    if absoluteValueError:
        return abs(hypothesis(point, weights) - point[-1])
    return hypothesis(point, weights) - point[-1]

def updateWeights(data, weights, alpha):
    newWeights = []
    for index, weight in enumerate(weights):
        sum = 0
        
        for point in data:
            try:
                sum += rawError(point, weights) * point[0]**index
            except (TypeError, IndexError):
                sum += rawError(point, weights) * point**index


        newWeights.append(weight - alpha * (1/len(data)) * sum)
    return newWeights

def overallError(data, weights):
    ans = pp.retrieveColumn(data, -1)
    guesses = [hypothesis(point, weights) for point in data]
    return mean_squared_error(ans, guesses)


def generateModel(data, weights, alpha, error, delta):
    #print(data)
    #visualize(data, weights)
    #alpha = 0.01
    while((err := abs(overallError(data, weights))) > error):
        #print(overallError(data, weights))
        weights = updateWeights(data, weights, alpha)
        #visualize(data, weights)
        #print(err, abs(overallError(data, weights)), alpha)
        if abs((newErr := abs(overallError(data, weights))) - err) < delta: 
            #visualize(data, weights)
            return weights, newErr
        elif newErr > err:
            alpha *= .01
        else:
            alpha *= 1.5
    #visualize(data, weights)
    return weights, err

#alpha = 0.001
newWeights = []
errors = []
errorBound = 50
benchmarks = [35, .5, 10, .5, 10, .5]

print(overallError(data[0], weights[0]))


test = updateWeights(data[0], weights[0], alpha)
print(overallError(data[0], test))

for index, weightSet in enumerate(weights):
    #if (dSet := index % 2) == 0:
    newWeightSet, error = generateModel(data[index % 2], weightSet, alpha, benchmarks[index], giveUp)
    newWeights.append(newWeightSet)
    errors.append(error)
    #else:


titles = ["synthetic-1.csv", "synthetic-2.csv"]
for index, weightSet in enumerate(newWeights):
    print(weightSet)
    print(errors[index])
    visualize(data[index%2], weightSet, titles[index%2])
