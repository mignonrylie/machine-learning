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

#working
def disassemble(data):
    columns = []
    #for each value in point (for each column)
    for col in range(len(data[0])):
        column = []
        for index in range(len(data)):
            column.append(data[index][col])
        columns.append(column)


    print(columns)
    return columns

def reassemble(data):
    reassembled = []
    for index in range(len(data[0])):
        point = []
        for column in data:
            point.append(column[index])
        reassembled.append(point)
    return reassembled

def standardize(data):
    data = disassemble(data)

    standard = []
    for column in data:
        xbar = sum(column) / len(column)
        sd = np.std(column)
        standardCol = []
        for value in column:
            std = (value - xbar) / sd
            standardCol.append(std)
        standard.append(standardCol)

    return reassemble(standard)

def squish(data):
    data = disassemble(data)

    squished = []
    for column in data:
        maxVal = max(column)
        minVal = min(column)
        squishedCol = []
        for point in column:
            squish = (point - minVal) / (maxVal - minVal)
            squishedCol.append(squish)
        squished.append(squishedCol)

    return reassemble(squished)
    