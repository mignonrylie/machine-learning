import tree
import matplotlib.pyplot as plt
import numpy as np

files = ["synthetic-1-discrete.csv", "synthetic-2-discrete.csv", "synthetic-3-discrete.csv", "synthetic-4-discrete.csv"]
trees = []
datas = []

for index, file in enumerate(files):
    data = None
    root = None
    data, nontitles = tree.importData(file)
    root = tree.id3(data, -1, [0,1])
    datas.append(data)
    trees.append(root)

for index, data in enumerate(datas):
    best, acc = tree.generateOptimalTree(data, 5)
    print(files[index])
    print("best accuracy is : " + str(acc))
    print(tree.RenderTree(best))
    print()

poke, labels = tree.importData("pokemonStats-discrete.csv")
pokebest, acc = tree.generateOptimalTree(poke, 5)
print("pokemonStats-discrete.csv")
print("best accuracy is : " + str(acc))
print(tree.RenderTree(pokebest))

#visualization:
def visualize(data, givenTree, idx):
    zeros = [point for point in data if point[-1] == '0']
    ones = [point for point in data if point[-1] == '1']

    zeros = tree.uniqueList(zeros)
    ones = tree.uniqueList(ones)

    x0 = [int(point[0]) for point in zeros]
    y0 = [int(point[1]) for point in zeros]
    x1 = [int(point[0]) for point in ones]
    y1 = [int(point[1]) for point in ones]

    fig, ax = plt.subplots()
    fig.suptitle(idx)
    
    ax.set_xlabel("Attribute 0")
    ax.set_ylabel("Attribute 1")

    plt.xticks(np.arange(0, 5, 1.0))
    plt.yticks(np.arange(0, 5, 1.0))

    a = [0, 1, 2, 3, 4]
    b = [0, 1, 2, 3, 4]
    z = np.random.rand(5, 5)
    for indexa, ai in enumerate(a):
        for indexb, bi in enumerate(b):
            z[indexb][indexa] = tree.predict(givenTree, [ai, bi])
        
    ax.pcolormesh(a, b, z)

    ax.scatter(x0, y0, c=[[1,1,1]], marker='$0$')
    ax.scatter(x1, y1, c=[[0,0,0]], marker='$1$')

    plt.show()

visualize(datas[0], trees[0], files[0])
visualize(datas[1], trees[1], files[1])
visualize(datas[2], trees[2], files[2])
visualize(datas[3], trees[3], files[3])