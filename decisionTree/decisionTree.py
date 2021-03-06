from importlib.machinery import SourcelessFileLoader
import math
import csv

#should already be discretized
#TODO: generalize for n columns
def importData(filename): 
    csvData = [[]]
    with open(filename, newline = '') as file:
        reader = csv.reader(file, delimiter = ',')
        
        for row in reader:
            #print(row)
            if row == []:
                continue
            
            dataPoint = []

            dataPoint.append(row[0])
            dataPoint.append(row[1])
            dataPoint.append(row[2])

            csvData.append(dataPoint)

    del csvData[0]
    return csvData
        
importedData = importData("synthetic-1-discrete.csv")

def entropy(data, toMatch):
    #pass in a list of just class labels?
    
    i = 0
    for row in data:
        if row[-1] == toMatch: #assumes class label is last in list
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

    #pass in a list of just class labels?

    sum = 0
    seenLabels = [data[0][-1]] #assumes class label is the last index
    sum += entropy(data, seenLabels[0])
    for row in data:
        seen = False
        for label in seenLabels:
            if row[-1] == label:
                seen = True
        if not seen:
            seenLabels.append(row[-1])
            sum += entropy(data, seenLabels[-1])

    return sum

def dataIntoLists(data):
    split = []

    for index in range(len(data[0])): #assuming each data point is the same length; should be!
        newList = []
        for row in data:
            newList.append(row[index])
        split.append(newList)

    return split

data = dataIntoLists(importedData)
print(data)

def splitValues(attribute, labels):
    seenValues = []
    values = [[]]

    for index, value in attribute:
        #temp = [[value], [classlabel]]

        #if value has been seen:
            #values[value][0]

        #if value has not been seen:




        pass


#calculates information gain for one attribute
def informationGain(col, data): #given as column number of dataset, data being the entire data set
    attribute = data[col]
    labels = data[-1] #assumes labels are the end of the data

    #must use entire data set since that's how entropy function is written
    #may want to rewrite entropy function to just work with the class labels?

    #entropy(data) - sum for each value of (number of points with that value/total points)*entropy(points with that value)

    #for each value subset, the class label must be attached.
    #meaning that we shouldn't separate the class label from the data.

    #if i do choose to split the data into each attribute, it should also have the class label
    #we may want to instead keep a copy of the class label with every attribute list

    #two ways to lay this out:
    #[[val1, val2, label], [val1, val2, label], [val1, val2, label], ...] think I'm gonna go with this one
        #then split attributes as:
        #[[[val1, label], [val1, label], ...] [[val2, label], [val2, label], ...]]
    #[[val1, val1, val1, ...], [val2, val2, val2, ...], [label, label, label, ...]]

    #need number of values that the attribute can have 
    #need to generate data subsets for each value


    #inside sum, for each value:
    #for each value, (number of points with that value/total points)*entropy(points with that value)

    #each value will have

    #count values for the attribute:
    #for each attribute:
        #if it's a value we've already seen
            #add the value+label

    pass









#positive = 1
#negative = 0

#ID3(Examples, Target_Attribute, Attributes)
#	Create a root node for the tree
#	If all examples are 1, Return the single-node tree Root, with label = 1
#	If all examples are 0, Return the single-node tree Root, with label = 0
#	If number of predicting attributes is empty, then Return the single node tree Root, with label = most common value of the target attribute in the examples
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

#at each iteration split into [[target_attribute], [other_attributes], [class_label]]
#examples = class labels

def findMajority(data, col):
    pass

class Node:
    def __init__(self, attribute, data): #self is not passed in. #how should attribute be represented? column number?
        self.left = None
        self.right = None
        self.attribute = None #attribute that we're splitting on. is null in the leaf.
        self.label = None #label that is assigned. is null in internal nodes.
        self.data = data #copy of all the data (minus the attributes we've already split on?)

        self.classLabels = []
        self.allOne = True
        self.allZero = True
        for point in data:
            self.classLabels.append(point[-1]) #assumes class labels are the final column of data
            if point[-1] == 0:
                self.allOne = False
            elif point[-1] == 1: #this if is redundant for binary class labels
                self.allZero = False


    def id3(self):
        #self.left and self.right assigned later
        if self.allOne:
            self.label = 1
            return self
        elif self.allZero:
            self.label = 0
            return self
        else:
            pass