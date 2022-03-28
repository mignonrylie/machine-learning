import csv
import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#toggle between binary and multiclass classification
binary = False

def importData(filename):
    data = []
    with open(filename, newline = '') as file:
        reader = csv.reader(file, delimiter = ',')
        for row in reader:
            point = []
            for value in row:
                point.append(int(value))
            data.append(point)
    return np.array(data)

#implemented here as sigmoid
def g(x):
    return 1 / (1 + np.power(math.e, -x))

#given an output array, this finds the highest activation, i.e. my model's most confident answer
def decideNumber(outputActivation):
    max = 0
    index = -1
    for idx, point in enumerate(outputActivation[0]):
        if point > max:
            max = point
            index = idx
    return index

#the class label is given as a single number, but it needs to look like an array of activations for my model
def classLabelToVector(expected, out):
    vector = []
    for index in range(len(out.flatten())):
        if expected == index:
            vector.append(1)
        else:
            vector.append(0)
    return vector

#feeds input through the whole network
def feedForward(inputs, wxh, who):
    hidden = g(np.dot(inputs, wxh.T))
    out = g(np.dot(hidden, who.T))
    return hidden, out

#finds the difference between the actual and the target output activation
def predictionError(expected, predicted):
    return classLabelToVector(expected, predicted) - predicted

def calculateDeltas(out, error, hidden, who):
    deltao = out * (1 - out) * error
    deltah = hidden * (1 - hidden) * np.dot(deltao, who)
    return deltao, deltah

def updateWeights(who, alpha, deltao, hidden, wxh, deltah, inputs):
    new_who = who + alpha * deltao.T * hidden
    new_wxh = wxh + alpha * deltah.T * inputs
    return new_who, new_wxh

def outputToClassLabel(out):
    max_val = 0
    index = 0
    out = out.flatten()
    for idx, point in enumerate(out):
        if point > max_val:
            max_val = point
            index = idx
    return index

def accuracy(data, wxh, who):
    correct = 0
    incorrect = 0
    for sample in data:
        point = np.array([sample[1:]])
        classLabel = sample[0]

        hidden, out = feedForward(point, wxh, who)
        guessLabel = outputToClassLabel(out)

        if guessLabel == classLabel:
            correct += 1
        else:
            incorrect += 1

    return correct, incorrect

if binary:
    data = importData("data/mnist_train_0_1.csv")
    who = np.random.uniform(-1, 1, size=(2, 28))
    out = np.array([range(2)])
    testData = importData("data/mnist_test_0_1.csv")

else:
    data = importData("data/mnist_train_0_4.csv")
    who = np.random.uniform(-1, 1, size=(5, 28))
    out = np.array([range(5)])
    testData = importData("data/mnist_test_0_4.csv")

wxh = np.random.uniform(-1, 1, size=(28, 784))
hs = np.array([range(28)])
bh = np.array([0.5 for x in hs])
bo = np.array([0.5 for x in out])
alpha = 0.01

#initial accuracy
print(accuracy(data, wxh, who)[0] / len(data))

samples = data
epochs = 0
ratio = 0
while epochs < 100 and (1 - ratio) > 0.05:
    epochs += 1
    np.random.shuffle(samples)
    for sample in samples:
        xs = np.array([sample[1:]])
        hs, out = feedForward(xs, wxh, who)
        error = predictionError(sample[0], out)
        deltao, deltah = calculateDeltas(out, error, hs, who)
        who, wxh = updateWeights(who, alpha, deltao, hs, wxh, deltah, xs)
    correct = accuracy(data, wxh, who)[0]
    ratio = correct / len(data)
    print(ratio)

print("final accuracy:")
print(accuracy(testData, wxh, who)[0] / len(testData))

#visualization, just for fun
def showDrawing(sample, prediction):
    pixels = sample[1:]
    pixels = [255 - x for x in pixels]
    pixels = np.reshape(pixels, (28, 28))
    fig = plt.figure()
    ax = fig.add_subplot()
    imgplot = plt.imshow(pixels, cmap = 'gray')
    ax.set_title('Prediction: ' + str(prediction))
    plt.show()

#show the image and print my model's guess
samples = np.random.choice(len(testData), 10)
for i in samples:
    hs, out = feedForward(testData[i][1:], wxh, who)
    prediction = outputToClassLabel(out)
    showDrawing(testData[i], prediction)