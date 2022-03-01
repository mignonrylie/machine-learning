#may want to do feature scaling - normalization?
import csv

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