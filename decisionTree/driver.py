from posixpath import split
from re import U
from tkinter import Y
import decisionTree2 as tree
#from anytree import Node, RenderTree
import anytree

import matplotlib.pyplot as plt
import numpy as np

data = tree.importData("synthetic-4-discrete.csv")

root = tree.id_3(data, -1, [0, 1], None, None)

print(anytree.RenderTree(root))




splitvalues = [node.value for node in anytree.PreOrderIter(root)]
print(splitvalues)
splitattributes = [node.attribute for node in anytree.PreOrderIter(root)]
print(splitattributes)

#the attribute decides whether the line will be horizontal or vertical


zeroes = []
ones = []

for point in data:
    if point[-1] == 0:
        zeroes.append(point)
    if point[-1] == 1:
        ones.append(point)


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

plt.xticks(np.arange(min(data[0])-1, max(data[0])+1, 1.0))
plt.yticks(np.arange(min(data[1])-1, max(data[1])+1, 1.0))

plt.show()