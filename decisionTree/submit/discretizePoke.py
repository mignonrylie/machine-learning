import csv

data = []
split = 5 #3-5 splits should be fine

features = []
labels = []
intervals = [] #list of numbers
mins = []
boundaries = []
discreteFeatures = []
fullyDiscrete = []

def loadData(filename):
    data = []
    
    with open(filename, newline = '') as file:
        reader = csv.reader(file, delimiter = ',')
        for index, row in enumerate(reader):
            dataPoint = []

            for point in row:
                if index == 0:
                    dataPoint.append(point)
                else:
                    dataPoint.append(int(point))
                    
            if index == 0:
                labels.append(dataPoint)
            else:
                data.append(dataPoint)
            
    return data

data = loadData("pokemonStats.csv")

def splitColumns(data):
    features = []

    for col in range(len(data[0])):
        column = []
        for row in data:
            column.append(row[col])
        features.append(column)

    return features
    
features = splitColumns(data)

def findInterval(col):
    max = col[0]
    min = col[0]

    for i in range(len(col)):
        if col[i] > max:
            max = col[i]

        if col[i] < min:
            min = col[i]

    if max == 1 and min == 0:
        return -1, -1 #indicates boolean, don't need an interval for that
    else:
        return (max - min) / split, min

for col in features:
    interval, minimum = findInterval(col)
    intervals.append(interval)
    mins.append(minimum)

#boolean boundaries are empty lists
def valueBoundaries(interval, min):
    boundaries = []
    if interval == -1 and min == -1:
        return boundaries
    for i in range(split-1):
        boundaries.append(min + (i+1)*interval)

    return boundaries

for index, col in enumerate(intervals):
    boundaries.append(valueBoundaries(col, mins[index]))

def discretize(feature, boundary):
    discrete = []
    for i in range(len(feature)):
        if len(boundary) == 0:
            discrete.append(feature[i])
        else:
            for j in range(len(boundary)):
                if feature[i] < boundary[j]:
                    discrete.append(j)
                    break
                elif j == len(boundary)-1 and feature[i] >= boundary[j]: #accounts for feature values that belong in the last bin
                    discrete.append(len(boundary))
                else:
                    pass

    return discrete

for index, feature in enumerate(features):
    discreteFeatures.append(discretize(feature, boundaries[index]))

def discretizeLabel(label):
    discrete = []

    for i in range(len(label)):
        discrete.append(int(label[i]))

    return discrete

#fixed i think! :)
def reassemble(discreteFeatures):
    assembled = labels

    for index, col in enumerate(discreteFeatures[0]):
        point = []
        for row in discreteFeatures:
            point.append(row[index])
        assembled.append(point)

    return assembled

fullyDiscrete = reassemble(discreteFeatures)

def writeCSV(filename, data):
    with open(filename, 'w', newline = '') as file:
        writer = csv.writer(file, delimiter = ',')
        for row in data:
            writer.writerow(row)
        #for i in range(len(data)):
        #    writer.writerow(data[i])

writeCSV("pokemonStats-discrete.csv", fullyDiscrete)