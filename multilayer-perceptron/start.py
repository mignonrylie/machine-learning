import csv
import math
import numpy as np

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

#ONLY VALID WHEN g() IS SIGMOID!!
def gPrime(x):
    return g(x) * (1 - g(x))

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







#randomly initialize weights
hiddenWeights = np.random.uniform(-1, 1, size=(785, 28))
if zeroToFour:
    outputWeights = np.random.uniform(-1, 1, size=(28, 5))
else:
    outputWeights = np.random.uniform(-1, 1, size=(28, 2))



#feed forward an input through the NN, 'making a guess' 
def feedForward(point):
    #feed into hidden layer
    hiddenInputs = np.dot(hiddenWeights.T, point)
    #output of hidden layer's activation
    hiddenOut = g(hiddenInputs)

    #feed into output layer
    finalInputs = np.dot(outputWeights.T, hiddenOut)
    #output of output layer's activation
    finalOut = g(finalInputs)

    return hiddenInputs, finalInputs, finalOut

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



data = importData("data/mnist_train_0_1.csv")
initX = data[0]
#TODO: bias???
initX[0] = 1

#finalInputs is the hidden activations times the corresponding weights. 
hiddenInputs, finalInputs, outputNeurons = feedForward(initX)



print(outputNeurons)
err = getError(outputNeurons, data[0])
print(err)

alpha = 0.1

#updating weights
updatedWeights = []
exit()

point = data[0]

#used here:
#final (output neurons) - 2x1
#outputWeights (weights between hidden and output layer) - 28x2
for neuron, index in enumerate(outputNeurons):
    for weight in outputWeights[index]:
        #for weight associated with neuron:
        #newWeight = weight + alpha * err * gPrime(in) * Xj
        #where in is the input value for the neuron in question
        # and Xj is 
        #point is the data point of comparison
        newWeight = weight + alpha * err * gPrime(finalInputs[index]) * point[index]