
del data[0]
del data[len(data)-1]

def findInterval(data, col):
    max = data[0][col]
    min = data[0][col]

    for point in data:
        if point[col] > max:
            max = point[col]

        if point[col] < min:
            min = point[col]

    return (max - min) / split

def splitFeatures(data, col):
    feature = []

    for i in range(len(data)):
        feature.append(data[i][col])

    return feature

feature1 = splitFeatures(data, 0)
feature2 = splitFeatures(data, 1)
classLabels = splitFeatures(data, 2)

interval1 = findInterval(data, 0) #interval for first feature
interval2 = findInterval(data, 1) #interval for second feature
#don't need an interval for the last column - that's the class label

print(interval1, interval2)

def valueBoundaries(interval):
    boundaries = []
    for i in range(split-1):
        boundaries.append((i+1)*interval)

    return boundaries

boundary1 = valueBoundaries(interval1)
boundary2 = valueBoundaries(interval2)

def discretize(feature, boundary):
    discrete = []
    for i in range(len(feature)):
        for j in range(len(boundary)):
            if feature[i] >= boundary[j]:
                discrete.append(j)
            else:
                pass

    return discrete

discrete1 = discretize(feature1, boundary1)
discrete2 = discretize(feature2, boundary2)

for i in range(len(discrete2)):
    print(discrete2[i])

print(len(discrete1))
print(len(discrete2))
print(len(classLabels))

def reassemble(col1, col2, col3):
    assembled = [[]]

    for i in range(len(col1)):
        point = []
        point.append(col1[i])
        point.append(col2[i])
        point.append(col3[i])
        assembled.append(point)

    return assembled


discreteData = reassemble(discrete1, discrete2, classLabels)

for i in range(len(discreteData)):
    print(discreteData[i])