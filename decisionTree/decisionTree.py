import math
import csv

#ID3(Examples, Target_Attribute, Attributes)
#	Create a root node for the tree
#	If all examples are positive, Return the single-node tree Root, with label = +
#	If all examples are negative, Return the single-node tree Root, with label = -
#	If number of predicting attributes is empty, then Return the single node tree Root,
#		,with label = most common value of the target attribute in the examples
#	Else Begin
#		A <- The Attribute that best classifies examples (this is information gain)
#		Decision Tree attribute for Root = A
#		For each possible value vi of A
#			Add a new tree branch below Root, corresponding to the test A = vi
#			Let Examples[vi] be the subset of examples that have the value vi for A
#			If Examples[vi] is empty
#				Below this new branch add a leaf node with label = most common target value in the examples
#			Else below this new branch add the subtree ID3(Examples[vi], Target_Attribute, Attributes - {A})
#	End
#Return Root

#should already be discretized
def importData(filename): 
    csvData = [[]]
    with open(filename, newline = '') as file:
        reader = csv.reader(file, delimiter = ',')
        for row in reader:
            dataPoint = []

            dataPoint.append(row[0])
            dataPoint.append(row[1])
            dataPoint.append(row[2])

            csvData.append(dataPoint)

    return csvData
        
def entropy(data, toMatch):
    #go through each data point
    #if it's a new class label, that's a new entry in labels and a new count
    #c
    
    i = 0
    for row in data:
        if row[-1] == toMatch:
            i += 1

    prob = i/len(data)

    return prob*math.log(prob, 2)


#assume given a list of data points with class labels
def calculateEntropy(data):
    #dictionary structure:
    #occurances = {classLabel : count, cl2 : c2, cl3 : c3, ...}
    #for row in data:
    # if classLabel is not in the dictionary, add it to the dictionary with key 0.
    # remove the corresponding classLabel's key and replace it with the value incremented by one

    sum = 0
    seenLabels = [data[0][-1]] #assumes class label is the last index
    sum += entropy(data, seenLabels[0])
    for row in data:
        seen = False
        for label in seenLabels:
            if row[-1] == seenLabels:
                seen = True
        if not seen:
            seenLabels.append(row[-1])
            sum += entropy(data, seenLabels[-1])

    return sum
