import csv

data = []
split = 5 #3-5 splits should be fine

f1 = []
f2 = []
labels = []

df1 = []
df2 = []
dlabels = []

#one for each non-discrete feature value
feature1 = []
feature2 = []
interval1 = 0
interval2 = 0
boundary1 = []
boundary2 = []
discrete1 = []
discrete2 = []

discreteData = []

def loadData(filename):
    data = [[]]
    with open(filename, newline = '') as file:
        reader = csv.reader(file, delimiter = ',')
        for row in reader:
            dataPoint = []

            
            dataPoint.append(float(row[0]))
            dataPoint.append(float(row[1]))
            dataPoint.append(row[2])

            data.append(dataPoint)

    return data

data = loadData("synthetic-4.csv")

del data[0]

def splitColumns():
    for i in range(len(data)):
        f1.append(data[i][0])
        f2.append(data[i][1])
        labels.append(data[i][2])

splitColumns()

def findInterval(col):
    max = col[0]
    min = col[0]

    for i in range(len(col)):
        if col[i] > max:
            max = col[i]

        if col[i] < min:
            min = col[i]

    return (max - min) / split, max, min

interval1, max1, min1 = findInterval(f1)
interval2, max2, min2 = findInterval(f2)

print(min1, max1, interval1)


def valueBoundaries(interval, min):
    boundaries = []
    for i in range(split-1):
        boundaries.append(min + (i+1)*interval)

    return boundaries

boundary1 = valueBoundaries(interval1, min1)
boundary2 = valueBoundaries(interval2, min2)

print(boundary1)

def discretize(feature, boundary):
    discrete = []
    for i in range(len(feature)):
        for j in range(len(boundary)):
            if feature[i] < boundary[j]:
                discrete.append(j)
                break
            elif j == len(boundary)-1 and feature[i] >= boundary[j]: #accounts for feature values that belong in the last bin
                discrete.append(len(boundary))
            else:
                pass

    return discrete

def discretizeLabel(label):
    discrete = []
    for i in range(len(label)):
        discrete.append(int(label[i]))

    return discrete

print(len(f1), len(f2), len(labels))

df1 = discretize(f1, boundary1)
df2 = discretize(f2, boundary2)
dlabels = discretizeLabel(labels)

print(len(df1), len(df2), len(dlabels))

print(df1)

def reassemble(col1, col2, col3):
    assembled = [[]]

    for i in range(len(col1)):
        point = []
        point.append(col1[i])
        point.append(col2[i])
        point.append(col3[i])
        assembled.append(point)

    return assembled

discreteData = reassemble(df1, df2, labels)

def writeCSV(filename):
    with open(filename, 'w', newline = '') as file:
        writer = csv.writer(file, delimiter = ',')
        for i in range(len(discreteData)):
            writer.writerow(discreteData[i])

writeCSV("synthetic-4-discrete.csv")