from cmath import inf
import csv
import math
from anytree import Node, RenderTree
import inspect

import matplotlib.pyplot as plt
import numpy as np
#[[v1, v2, label], [v1, v2, label], ...]

#data is read in correctly
#id3 > bestAttribute > informationGain > entropy > colToList(wrong)


#TODO: turn attribute labels into enumeration
#TODO: visualization

def importData(filename): 
    csvData = [[]]
    with open(filename, newline = '') as file:
        reader = csv.reader(file, delimiter = ',')
        
        for row in reader:
            if row == []:
                continue
            
            dataPoint = []
            dataPoint.append(int(row[0]))
            dataPoint.append(int(row[1]))
            dataPoint.append(int(row[2]))

            csvData.append(dataPoint)

    del csvData[0]
    return csvData
        
importedData = importData("synthetic-1-discrete.csv")

def colToList(given, index):
    newList = []
    for row in given:
        newList.append(row[index])

    return newList

def removeCol(given, index):
    newList = []
    for row in given:
        point = []
        for idx, entry in enumerate(row):
            if idx != index:
                point.append(entry)
        newList.append(point)

    return newList


def uniqueList(given):
    unique = []

    for i in given:
        seen = False
        for j in unique:
            if i == j:
                seen = True

        if not seen:
            unique.append(i)

    return unique

def entropy(data): #assumes class label is last in row
    labels = colToList(data, -1)
    labels = uniqueList(labels)

    sum = 0
    for label in labels:
        num = 0
        for i in data:
            if i[-1] == label:
                num += 1

        prob = num/len(data)
        if prob == 0:
            log = 0
        else:
            log = math.log2(prob)

        sum += -prob*log

    return sum

def informationGain(attribute, data): #attribute as index
    values = colToList(data, attribute)
    toCheck = uniqueList(values)

    assemble = []

    for index, val in enumerate(values):
        point = []
        point.append(val)
        point.append(data[index][-1])#pass in the class values as well
        assemble.append(point)

    values = assemble

    sum = 0
    for label in toCheck:
        subset = []
        for val in values:
            if val == label:
                subset.append(val)
        
        frac = len(subset)/len(values)
        sum += frac*entropy(subset)

    return entropy(data) - sum

def bestAttribute(data, cols): #again, assuming class label is index -1

    maxGain = -1
    bestAttribute = None
    for attribute in cols: #iterate through columns (attributes) except class label range(len(data[0])-1)
        if informationGain(attribute, data) > maxGain:
            bestAttribute = attribute
    return bestAttribute

def allOne(data): #class label as -1
    for row in data:
        if row[-1] != '1':
            return False
    return True

def allZero(data): #class label as -1
    for row in data:
        if row[-1] != '0':
            return False
    return True

def mostCommon(data):
    print(inspect.stack()[1].function)
    print(data)
    labels = uniqueList(colToList(data, -1))
    print(colToList(data, -1))

    print(labels)
    mostCommon = None
    count = -1
    for label in labels:
        sum = 0
        for row in data:
            if data[-1] == label:
                sum += 1
        if sum > count:
            count = sum
            mostCommon = label
    print(mostCommon)
    return mostCommon

def returnSubset(data, attribute, value):
    subset = []
    for row in data:
        if row[attribute] == value:
            subset.append(row)
    return subset

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

#target = thing you're trying to predict = 

def id_3(examples, target_attribute, attributes, givenParent, value):
    if givenParent is not None:
        root = Node("root", parent = givenParent, depth = givenParent.depth + 1, value = value, attribute = None)
    else:
        root = Node("root", depth = 0, value = None, attribute = None)
    
    if allOne(examples):
        root.label = 1
        return root
    
    elif allZero(examples):
        root.label = 0
        return root

    elif attributes is None or len(attributes) == 0:
        root.label = mostCommon(examples)
        return root

    else:
        root.attribute = bestAttribute(examples, attributes)
        possibleValues = uniqueList(examples[root.attribute])

        print("att:")
        print(root.attribute)
        for value in possibleValues:
            subset = returnSubset(examples, root.attribute, value)
            if len(subset) == 0:
                id_3(subset, -1, None, root, value)

            else:
                id_3(subset, -1, [x for x in attributes if x != root.attribute], root, value)


        return root















