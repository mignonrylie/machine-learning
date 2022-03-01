import csv
from math import log2
from anytree import Node, RenderTree
import anytree
import matplotlib.pyplot as plt
import numpy as np

def importData(filename): 
    csvData = []
    titles = []
    with open(filename, newline = '') as file:
        reader = csv.reader(file, delimiter = ',')
        
        for index, row in enumerate(reader):
            dataPoint = []

            if filename == "pokemonStats-discrete-TEST.csv":
                if index == 0:
                    for point in row:
                        titles.append(point)
                else:
                    for point in row:
                        dataPoint.append(point)

            else:
                for point in row:
                    dataPoint.append(point)

            if dataPoint != []:
                csvData.append(dataPoint)

    titles = csvData[0]
    del csvData[0]

    return csvData, titles

def allSame(examples, target):
    same = True
    reference = examples[0][target]

    for example in examples:
        if example[target] is not reference:
            same = False

    return same

#expects single dimensional list
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

def mostCommon(examples, target):
    values = uniqueList([x[target] for x in examples])
    most = -1
    common = None

    for value in values:
        count = 0
        #count occurances of value in the data
        for point in examples:
            if point[target] == value:
                count += 1
        
        if count > most:
            most = count
            common = value

    return common
    
#make sure value is not passed in literally, always pull it from the data
def getSubset(examples, attribute, value):
    subset = []

    for example in examples:
        if example[attribute] == value:
            subset.append(example)

    return subset

def entropy(examples, target):
    labels = uniqueList([x[target] for x in examples])
    
    sum = 0
    for label in labels:
        subset = getSubset(examples, target, label)
        prob = len(subset)/len(examples)

        sum -= prob * log2(prob)

    return sum

def informationGain(examples, attribute, target):
    #entropy of the entire set minus the sum:
    sum = 0
    values = uniqueList([x[attribute] for x in examples])
    #for each possible value in the attribute, 
    for value in values:
        subset = getSubset(examples, attribute, value)
        ratio = len(subset)/len(examples)

        sum += ratio*entropy(subset, target)
    #find the ratio * entropy of the subset
    return entropy(examples, target) - sum

#examples is the data
#target is what you want to predict - in our case, the class label
#attributes is a list of attributes that are still good to use
#titles is for the csv column names, like in pokemonStats
def id3(examples, target, attributes,  givenParent = None, givenValue = None, titles = None):
    if givenParent is None:
        root = Node("", parent = None, depth = 0)
    else:
        root = Node("", parent = givenParent, depth = givenParent.depth + 1)

    #I wanted to initalize depth this way too, but I ran into 'can't set attribute' for some reason.
    if givenValue is not None:
        root.value = givenValue

    if root.depth == 2:
        root.label = mostCommon(examples, target)
        return root
    
    if len(examples) == 0:
        #return leaf with most common label in the parent node's set of examples
        root.label = mostCommon(root.parent.examples, root.parent.target)
        return root

    #if all labels same
    elif allSame(examples, target):
        #return leaf with that label
        root.label = examples[0][target]
        return root

    #elif no attributes left
    elif len(attributes) == 0:
        #return leaf with most common label
        root.label = mostCommon(examples, target)
        return root

    else:
        #pick the attribute that has the highest information gain
        bestAttribute = None
        highestGain = -1
        for attribute in attributes:
            gain = informationGain(examples, attribute, target)
            if gain > highestGain:
                bestAttribute = attribute
                highestGain = gain

        #for each value of that attribute,
        root.attribute = bestAttribute #set the attribute that we're splitting on
        values = uniqueList([x[bestAttribute] for x in examples])
        if titles is not None:
            root.title = titles[root.attribute]

        for value in values:
            subset = getSubset(examples, bestAttribute, value)
            id3(subset, target, [x for x in attributes if x != bestAttribute], root, value, titles)
            #create a child; its examples will be the subset of parent's examples with that value


    return root


def predict(tree, example):
    #follow the tree:
    #look at the attribute: whatever that value is in the example, go to the corresponding child
    #if there is no attribute (no child), get the value of the example, and return the corresponding label
    while tree.is_leaf is False: #only nodes with children should have attributes
        try:
            attribute = tree.attribute
            value = example[attribute]
            found = False
            for child in tree.children:
                if child.value == value:
                    tree = child
                    found = True

            #if we reach this point, no value matched. in this case we just assign it to the closest value
            if not found:
                difference = 10000
                closestChild = tree.children[0]
                for child in tree.children:
                    if abs(float(int(child.value)) - float(value)) < difference:
                        difference = abs(float(child.value) - float(value))
                        closestChild = child
                tree = closestChild

        except AttributeError: #hopefully redundant
            break

    return tree.label

def evaluate(tree, example):
    if predict(tree, example) == example[-1]:
        return True
    return False

def treeAccuracy(tree, data):
    correct = 0
    incorrect = 0 #redundant
    for example in data:
        if evaluate(tree, example):
            correct += 1
        else:
            incorrect += 1
    if correct + incorrect != len(data):
        print("something has gone wrong! wrong num of examples?")
        return None
    return correct / len(data)

def fold(data, k):
    #set aside 1/k of the examples for testing
    #the rest are training
    subsets = []
    num = len(data) // k
    remainder = len(data) % k
    
    for i in range(k): #reserve the last fold for testing
        begin = i*num
        end = (i+1)*num
        fold = []
        for j in range(begin, end):
            fold.append(data[j])
        subsets.append(fold)

    if remainder != 0: #distributes remainders across sets
        count = 0;
        for i in range(num*k, num*k+remainder):
            subsets[count%k].append(data[i])
            count += 1

    return subsets

def kFolds(data, k):
    subsets = fold(data, k)
    allsets = []
    
    for test in range(k):
        testing = []
        training = []
        ttset = []
        for i in range(k):
            if i == test:
                testing = subsets[i]
            else:
                training.extend(subsets[i])
        ttset.append(training)
        ttset.append(testing)
        allsets.append(ttset)

    return allsets

#for each k-fold
#generate a tree and find its accuracy
#pick best tree
def generateOptimalTree(data, k):
    folds = kFolds(data, k)
    bestAccuracy = 0.0
    bestTree = None
    for fold in folds:
        tree = id3(fold[0], -1, [x for x in range(len(data[1])-1)])
        accuracy = treeAccuracy(tree, fold[1])
        if accuracy > bestAccuracy:
            bestTree = tree
            bestAccuracy = accuracy

    return bestTree, bestAccuracy