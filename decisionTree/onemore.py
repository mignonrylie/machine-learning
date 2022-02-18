#ID3 (Examples, Target_Attribute, Attributes)
#    Create a root node for the tree
#    If all examples are positive, Return the single-node tree Root, with label = +.
#    If all examples are negative, Return the single-node tree Root, with label = -.
#    If number of predicting attributes is empty, then Return the single node tree Root,
#    with label = most common value of the target attribute in the examples.
#    Otherwise Begin
#        A ← The Attribute that best classifies examples.
#        Decision Tree attribute for Root = A.
#        For each possible value, vi, of A,
#            Add a new tree branch below Root, corresponding to the test A = vi.
#            Let Examples(vi) be the subset of examples that have the value vi for A
#            If Examples(vi) is empty
#                Then below this new branch add a leaf node with label = most common target value in the examples
#            Else below this new branch add the subtree ID3 (Examples(vi), Target_Attribute, Attributes – {A})
#    End
#    Return Root

import csv
from math import log2
from anytree import Node, RenderTree
import anytree
import matplotlib.pyplot as plt
import numpy as np

import decisionTree2 as tree

#working as expected
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

            csvData.append(dataPoint)

    if filename == "pokemonStats-discrete-TEST.csv":
        addLabels(csvData)
    return csvData, titles

def addLabels(data):
    with open("pokemonLegendary.csv", newline= '') as file:
        reader2 = csv.reader(file, delimiter = ',')


        for index, row in enumerate(reader2):
            if index != 0:
                data[index].append(eval(str(row)))
    



#ID3 (Examples, Target_Attribute, Attributes)
#    Create a root node for the tree
#    If all examples are positive, Return the single-node tree Root, with label = +.
#    If all examples are negative, Return the single-node tree Root, with label = -.
#    If number of predicting attributes is empty, then Return the single node tree Root,
#    with label = most common value of the target attribute in the examples.
#    Otherwise Begin
#        A ← The Attribute that best classifies examples.
#        Decision Tree attribute for Root = A.
#        For each possible value, vi, of A,
#            Add a new tree branch below Root, corresponding to the test A = vi.
#            Let Examples(vi) be the subset of examples that have the value vi for A
#            If Examples(vi) is empty
#                Then below this new branch add a leaf node with label = most common target value in the examples
#            Else below this new branch add the subtree ID3 (Examples(vi), Target_Attribute, Attributes – {A})
#    End
#    Return Root

#working as expected
def allSame(examples, target):
    same = True
    reference = examples[0][target]

    for example in examples:
        if example[target] is not reference:
            same = False

    return same

#expects single dimensional list; working
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

#working as expected
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
    
#working, but be careful - make sure value is not passed in literally, always pull it from the data
#easy to run into string vs int issues
def getSubset(examples, attribute, value):
    subset = []

    for example in examples:
        if example[attribute] == value:
            subset.append(example)

    return subset

#working
def entropy(examples, target):
    labels = uniqueList([x[target] for x in examples])
    
    sum = 0
    for label in labels:
        subset = getSubset(examples, target, label)
        prob = len(subset)/len(examples)

        sum -= prob * log2(prob)

    return sum
    
#working i believe?
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
    #root = Node("", parent = givenParent, depth = 0)
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
    
    



    #only internal nodes should have attributes
    #only leaves should have labels


    #elif no examples -- thinking this should come first?
    if len(examples) == 0:
        #return leaf with most common label in the parent node's set of examples
        root.label = mostCommon(root.parent.examples, root.parent.target)
        return root
        #this could be done in the parent

    #if all labels same
    elif allSame(examples, target): #good, working
        #return leaf with that label
        root.label = examples[0][target]
        return root

    #elif no attributes left
    elif len(attributes) == 0: #working hopefully
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
        #print(values)
        if titles is not None:
            root.title = titles[root.attribute]

        for value in values:
            subset = getSubset(examples, bestAttribute, value)
            id3(subset, target, [x for x in attributes if x != bestAttribute], root, value, titles)
            #create a child; its examples will be the subset of parent's examples with that value
            #child(examples(value), target)
            #here is where you would set most common label if you weren't going to pass on any examples

    return root

#############################################################3


#loaded = importData("synthetic-1-discrete.csv")

#test = [x for x in loaded if x[-1] == loaded[-1][-1]]

#root1 = id3(loaded, -1, [0, 1])

#fakeNode = Node("testing", examples = loaded, target = -1)

#root = id3([], -1, [0, 1], givenParent=fakeNode)

#print(root.label)

#print(subset(loaded, -1, loaded[-1][-1]))


def draw(data):
    zeroes = []
    ones = []

    for point in data:
        if point[-1] == '0':
            zeroes.append(point)
        if point[-1] == '1':
            ones.append(point)

    
    fig = None
    ax = None


    fig, ax = plt.subplots()

    #doesn't print duplicates of points. makes the graph cleaner
    zeroes = tree.uniqueList(zeroes)
    ones = tree.uniqueList(ones)

    xZeroes = tree.colToList(zeroes, 0)
    yZeroes = tree.colToList(zeroes, 1)
    xOnes = tree.colToList(ones, 0)
    yOnes = tree.colToList(ones, 1)

    ax.scatter(xZeroes, yZeroes, c=[[0,0,0]], marker='$0$')
    ax.scatter(xOnes, yOnes, c=[[0,0,0]], marker='$1$')
    ax.set_xlabel("Attribute 0")
    ax.set_ylabel("Attribute 1")

    plt.xticks(np.arange(int(min(data[0]))-1, int(max(data[0]))+1, 1.0))
    plt.yticks(np.arange(int(min(data[1]))-1, int(max(data[1]))+1, 1.0))

    plt.show()




#files = ["synthetic-1-discrete.csv", "synthetic-2-discrete.csv", "synthetic-3-discrete.csv", "synthetic-4-discrete.csv"]


#for index, file in enumerate(files):
#    data = None
#    root = None
#    data = importData(file)
#    root = id3(data, -1, [0,1])
#    print(file + ":")
#    print(RenderTree(root))
#    draw(data)

print("synthetic:")
nondata, nontitles = importData("synthetic-1-discrete.csv")
nonroot = id3(nondata, -1, [0,1])
print(RenderTree(nonroot))


print("poke:")
data, title = importData("pokemonStats-discrete-TEST.csv")
del data[0]
root = id3(data, -1, [x for x in range(len(data[0])-1)], titles=title)
print(RenderTree(root))

print(root.children)



#evaluating a test example:
#follow the tree:
    #look at the attribute: whatever that value is in the example, go to the corresponding child
    #if there is no attribute (no child), get the value of the example, and return the corresponding label

def testExample(tree, example):
    while(tree.attribute is not None):
        for child in tree.children:
            if child.value == example[tree.attribute]:
                tree = child

    