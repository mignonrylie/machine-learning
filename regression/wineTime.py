from dataclasses import dataclass
import preprocess as pp

data = pp.readCSV("data/winequality-red.csv")
titles = data[0]
del data[0]
data = pp.toNumbers(data)

weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def hypothesis(point, weights):
    sum = 0
    for index, weight in enumerate(weights):
        if index == 0:
            sum += weights[index] #bias at index 0
        else:
            sum += weights[index] * point[index-1]
    return sum

def rawError(point, weights):
    return hypothesis(point, weights) - point[-1]

def overallError(data, weights):
    sum = 0
    for point in data:
        sum += rawError(point, weights)**2
    return sum / len(data)

def updateWeights(data, weights, alpha):
    newWeights = []
    for index, weight in enumerate(weights):
        sum = 0
        for point in data:
            if index == 0 :
                sum += rawError(point, weights) 
            else:
                sum += rawError(point, weights) * point[index-1]
        newWeights.append(weight - alpha * (1/len(data)) * sum)
    return newWeights

def generateModel(data, weights, alpha, error, delta):
    while (err := abs(overallError(data, weights))) > error:
        weights = updateWeights(data, weights, alpha)
        if abs((newErr := abs(overallError(data, weights))) - err) < delta:
            return weights, newErr
        elif newErr > err:
            alpha *= 0.01
        else:
            alpha *= 1.5
    return weights, err

benchmark = 1.5
alpha = 1e-10
error = benchmark
giveUp = 1e-7

weights, err = generateModel(data, weights, alpha, error, giveUp)

print(weights)
print(err)