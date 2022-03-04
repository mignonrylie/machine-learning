#may want to do feature scaling - normalization?
import csv
import numpy as np

def readCSV(filename):
    data = []
    with open(filename, newline = '') as file:
        reader = csv.reader(file, delimiter = ',')
        for row in reader:
            point = []
            for col in row:
                point.append(col)
            data.append(row)
    return data

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
        #print(point   )
        column.append(point[col])
    return column

def reassemble(x, y):
    reassembled = []
    for index in range(len(x)):
        point = [x[index], y[index]]
        reassembled.append(point)
    return reassembled


#xi - xbar / sd(x)
def standardize(data):
    #print(data)
    x = retrieveColumn(data, 0)
    #x = data
    y = retrieveColumn(data, 1)
    
    #try:
    #print(x)
    #print(sum(x))
    xbar = sum(x) / len(x)
    ybar = sum(y) / len(y)
    #except:
    #    print(x)
    sdx = np.std(x)
    sdy = np.std(y)

    xs = []
    for point in x:
        xs.append((point - xbar) / sdx)

    ys = []
    for point in y:
        ys.append((point - ybar) / sdy)

    return reassemble(xs, ys)


#xi - min / max - min
def squish(data):
    x = retrieveColumn(data, 0)
    #x = data
    y = retrieveColumn(data, 1)

    minX = min(x)
    maxX = max(x)

    minY = min(y)
    maxY = max(y)

    xs = []
    for point in x:
        xs.append((point - minX)/(maxX - minX))

    ys = []
    for point in y:
        ys.append((point - minY)/(maxY - minY))

    return reassemble(xs, ys)