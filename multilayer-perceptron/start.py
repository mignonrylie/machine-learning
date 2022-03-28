from codecs import BufferedIncrementalEncoder
import csv
import math
import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimage

import PIL

zeroToFour = False

#things to consider

#data as numeric label followed by 784 values 0-255

#input layer: 784 neurons, represented as a 784x1 matrix
#hidden layer: 28 neurons (sqrt(784) feels good)
#activation function: sigmoid()
#output layer: 2 (or 5 for part 2)

#each neuron has a list of weights, a connection for each of the neurons in the previous layer
#it will also recieve an activation value from each neuron in the previous layer
#so the input is the sum of each activation times the corresponding weight

#we'll need the error of the whole system, which is just the difference between the predicted and actual values.
#since i plan on having multiple output neurons, i'll do this for each output.

#input: 785 neurons == X
#(hidden weights) weights between layers: 784x28 matrix == Wh
#hidden layer: 28 neurons == H
#(output weights) weights between layers: 28x2 // 28x5 matrix == Wo
#output: 2/5 neurons == Y

#so g(Wh^T * X) will be the activation values to be fed into the hidden layer
#g(Wo^T * H) will be the activation values to be fed into the output layer






#backpropogation:
#whatever delta you get at the output layer (gradient?)
#is fed backwards, multiplied by the layer's weights







#implemented here as sigmoid
def g(x):
    #return 1 / (1 + math.pow(math.e, -x))
    return 1 / (1 + math.e**(-x))
    #return 1 / (1 + np.power(math.e, -x))

"""
#may not use because order of operations or whatever
#ONLY VALID WHEN g() IS SIGMOID!!
def gPrime(x):
    return g(x) * (1 - g(x))
"""

def importData(filename):
    #data = np.array(0)
    data = []
    with open(filename, newline = '') as file:
        reader = csv.reader(file, delimiter = ',')
        for row in reader:
            #point = np.array(0)
            point = []
            for value in row:
                #np.append(point, value)
                point.append(int(value))
            #point = np.array(point)
            #np.append(data, point)
            data.append(point)
    return np.array(data)


"""
#takes all the activations times the weights to get that neuron's input value
#working!
def calculateIntoNeuron(activations, weights):
    return np.sum(np.dot(weights.T, activations))

def calculateError(output, expected):
    return expected - output

def newWeightUpdate(weight, alpha, prevActivation, gin, expected):
    error = calculateError(gin, expected)
    print(weight, alpha, error, gPrime(gin), prevActivation)
    #return weight + alpha * error * gPrime(gin) * myinput
    #hard coding gPrime in rather than using the function, for order of operations purposes
    return weight + alpha * error * gin * (1 - gin) * prevActivation #gPrime(gin) * myinput
"""

"""
#where input is the activations times weights
#delta
def modifiedError(error, input):
    return error * g(input) * (1 - g(input))
"""


"""
#where activation is the activation of the previous neuron 
#input is the summed activations times weights of the following neuron
#weight is the weight between these
def modifiedWeightUpdate(weight, alpha, activation, error, input):
    return weight + alpha * activation * modifiedError(error, input)
"""

"""
#where input is the summed activations times weights of the concerned neuron
#weights is all the weights connecting the concerned neuron to the next layer
#and same for deltas
def myDelta(input, weights, deltas):
    return g(input) * (1 - g(input)) * sum(weights*deltas)
"""

"""
#ERROR: weird matrix stuff possible - be careful!
def internalWeightUpdate(weight, alpha, activation, delta):
    #return weight + alpha * activation * delta
    return weight + alpha * np.dot(activation, delta)
"""

"""
def getInnerDelta(activation, weight, prevDelta):
    return activation * (1 - activation) * weight * prevDelta
    #return activation * (1 - activation) * np.dot(weight, prevDelta)
"""












"""
#feed forward an input through the NN, 'making a guess' 
def feedForward(point):
    #feed into hidden layer
    hiddenInputs = np.dot(hiddenWeights.T, point) #TRANSPOSE
    #hiddenInput = sum(hiddenInputs)
    #output of hidden layer's activation
    hiddenOut = g(hiddenInputs)
    #hiddenActivation = g(hiddenInput) #this may in fact be correct

    #feed into output layer
    finalInputs = np.dot(outputWeights.T, hiddenOut) #TRANSPOSE
    #output of output layer's activation
    finalOut = g(finalInputs)

    return hiddenInputs, hiddenOut, finalInputs, finalOut
"""


"""
#guess being the output layer
def getError(guess, point):
    #the index referred to by point should be the only index with a value of 0
    goalOutput = []
    for i in range(len(guess)):
        if i == point[0]:
            goalOutput.append(1)
        else:
            goalOutput.append(0)
    
    errors = []
    for i in range(len(guess)):
        errors.append(goalOutput[i] - guess[i])

    totalError = 0
    for i in range(len(guess)):
        totalError += errors[i]**2

    return totalError

"""













#randomly initialize weights
hiddenWeights = np.random.uniform(-1, 1, size=(785, 28))
if zeroToFour:
    outputWeights = np.random.uniform(-1, 1, size=(28, 5))
else:
    outputWeights = np.random.uniform(-1, 1, size=(28, 2))

data = importData("data/mnist_train_0_1.csv")
data = importData("data/mnist_train_0_4.csv")
initX = data[0]
#TODO: bias???
initX[0] = 1

#finalInputs is the hidden activations times the corresponding weights. 
#hiddenInputs, hiddenNeurons, finalInputs, outputNeurons = feedForward(initX)


#print(outputNeurons)
#print(outputWeights)
#err = getError(outputNeurons, data[0])
#print(err)
#print()

alpha = 0.001

#updating weights
updatedWeights = []


point = data[0]

"""
print(outputWeights)
print(outputWeights.shape)
print(hiddenWeights.shape)
print(outputNeurons.shape)
print()"""
#used here:
#final (output neurons) - 2x1
#outputWeights (weights between hidden and output layer) - 28x2













"""
def updateWeights(neurons, weights, inputs, alpha, err, point):
    updated = np.zeros((len(weights), len(neurons)))
    #if len(neurons) == 2:
        #print(weights)
    #updated = []
    for index, neuron in enumerate(neurons):
        #newWeights = []
        for idx, weight in enumerate(weights):
            if idx == 0:
                x = 1
            else:
                x = inputs[index]
                #gPrime(inputs[idx])?
                #is activation at neuron j of previous layer

                #inputs should be the activation from the previous layer
                #before multiplying by weights
            #newWeight = weight[index] + alpha * err * gPrime(inputs[index]*weight[index]) * x
            newWeight = weight[index] + alpha * x * err * g(inputs[index]*weight[index]) * (1 - g(inputs[index]*weight[index]))


            #if len(neurons) == 2:
                #print(weight)
                #print(newWeight)
                #print()
            #newWeights.append(newWeight)
            updated[idx][index] = newWeight
        #updated.append(newWeights)
    return updated
#TODO: what about X0 = 1?
"""


"""

hiddenInputs, hiddenNeurons, finalInputs, outputNeurons = feedForward(initX)
err = getError(outputNeurons, data[0])

def fullUpdate(outputNeurons, outputWeights, finalInputs, err, hiddenNeurons, hiddenWeights, hiddenInputs):
    outputWeights = updateWeights(outputNeurons, outputWeights, finalInputs, alpha, err, point)
    hiddenWeights = updateWeights(hiddenNeurons, hiddenWeights, hiddenInputs, alpha, err, point)

    hiddenInputs, hiddenNeurons, finalInputs, outputNeurons = feedForward(initX)

    print(outputNeurons)
    #print(outputWeights)
    err = getError(outputNeurons, data[0])
    print(err)
    print()



"""


#samples = random.sample(data)

def predict(inputs, weights, bias):
    return g(np.dot(inputs, weights.T) + bias)

def fullPredict(inputs, hiddenWeights, outputWeights, hiddenBiases, outputBiases):
    hiddenActivations = predict(inputs, hiddenWeights, hiddenBiases)
    output = predict(hiddenActivations, outputWeights, outputBiases)
    return output

def decideNumber(outputActivation):
    #outs = outputActivation.flatten() #(1, 2) to (2,)

    max = 0
    index = -1
    for idx, point in enumerate(outputActivation[0]): #outs:
        if point > max:
            max = point
            index = idx

    return index

#index of correct number
def isGuessCorrect(output, index):
    return output == index

    correct = True
    for digit, idx in enumerate(output):
        if idx == index: #that digit should be 1
            if digit != 1:
                correct = False
        else: #that digit should be 0
            if digit != 0:
                correct = False
    return correct


def update(xs, wxh, hs, who, expected, out, bh, bo):
    hs = predict(xs, wxh, bh)
    out = predict(hs, who, bo)

    error = np.sum(expected - out)

    deltao = out * (1 - out) * error
    deltah = hs * (1 - hs) * np.dot(deltao, who)

    who = who + alpha * deltao.T * hs
    bo = bo + alpha * deltao
    wxh = wxh + alpha * deltah.T * xs
    bh = bh + alpha * deltah

    return hs, out, wxh, who, error, bh, bo

def errorWholeSet(samples, hiddenWeights, outputWeights, hiddenBiases, outputBiases):
    correct = 0
    incorrect = 0
    for sample in samples:
        input = sample[1:]
        prediction = fullPredict(input, hiddenWeights, outputWeights, hiddenBiases, outputBiases)
        output = decideNumber(prediction)
        if output == sample[0]:
            correct += 1
        else:
            incorrect += 1
    return (correct), (incorrect)

xs = np.array([data[0][1:]])
wxh = np.random.uniform(-1, 1, size=(28, 784))
hs = np.array([range(28)])
bh = np.array([0.5 for x in hs])
who = np.random.uniform(-1, 1, size=(2, 28))
#who = np.random.uniform(-1, 1, size=(5, 28))
out = np.array([range(2)])
#out = np.array([range(5)])
bo = np.array([0.5 for x in out])
#expected = np.zeros((1, 2)) #this can be up to 5
#expected = np.zeros((1, 5)) #this can be up to 5
#expected[0][data[0][0]] = 1
alpha = 0.01
error = 0


#xs = np.array([-3, 2, -1]) #test
#wxh = np.array([[-2, 0.5], [0.5, -1.5], [0.75, 0.6]])
#hs = np.array([])
#who = np.array([-0.25, 0.1])
#bh = 0
#bh = 0
sample = data[0]

#hs, out, wxh, who, error, bh, bo = update(xs, wxh, hs, who, sample[0], predict, bh, bo)

print(hs, out, wxh, who, error, bh, bo)

count = 0
error = 1

#xs = np.array([data[count][1:]])



samples = data
epochs = 0
incorrectRatio = 1
while incorrectRatio > 0.05 and epochs < 20:
    epochs += 1
    np.random.shuffle(samples)
    for sample in samples:
        xs = np.array([sample[1:]])
        out = fullPredict(xs, wxh, who, bh, bo)
        hs, out, wxh, who, error, bh, bo = update(xs, wxh, hs, who, sample[0], out, bh, bo)
        
    numcorrect, numincorrect = errorWholeSet(samples, wxh, who, bh, bo)
    incorrectRatio = numincorrect / len(samples)
    print(numcorrect/len(samples), numincorrect/len(samples))
    print()


testData = importData("data/mnist_test_0_1.csv")
#testData = importData("data/mnist_test_0_4.csv")

print("Test error:")
numcorrectTest, numincorrectTest = errorWholeSet(testData, wxh, who, bh, bo)
print(numincorrectTest)
print(numcorrectTest)




def showDrawing(sample):
    pixels = sample[1:]
    pixels = [255 - x for x in pixels]
    pixels = np.reshape(pixels, (28, 28))
    image = PIL.Image.fromarray(pixels.astype(np.uint8))
    image.show()


for i in range(10):
    prediction = fullPredict(testData[i][1:], wxh, who, bh, bo)
    finaloutput = decideNumber(prediction)
    print("Prediction:")
    #for x, idx in enumerate(finaloutput):
    #    if x == 0:
    #        print(idx)
    print(finaloutput)
    showDrawing(testData[i])
    input("hit enter for next guess:")

