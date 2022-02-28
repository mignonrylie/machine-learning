import onemore as driver
import matplotlib.pyplot as plt
import numpy as np

files = ["synthetic-1-discrete.csv", "synthetic-2-discrete.csv", "synthetic-3-discrete.csv", "synthetic-4-discrete.csv"]
trees = []
datas = []

for index, file in enumerate(files):
    data = None
    root = None
    data, nontitles = driver.importData(file)
    root = driver.id3(data, -1, [0,1])
    datas.append(data)
    trees.append(root)

#print(datas[0])
for data in datas:
    best, acc = driver.generateOptimalTree(data, 5)
    print("best accuracy is : " + str(acc))
    print(driver.RenderTree(best))
    print()


poke, labels = driver.importData("pokemonStats-discrete-TEST.csv")
#folds = fold(poke, 5)
#print(folds[0])
#print(poke[0])
pokebest, acc = driver.generateOptimalTree(poke, 5)
print("best accuracy is : " + str(acc))
print(driver.RenderTree(pokebest))



#visualization:
def visualize(data, tree, idx):
    zeros = [point for point in data if point[-1] == '0']
    ones = [point for point in data if point[-1] == '1']

    zeros = driver.uniqueList(zeros)
    ones = driver.uniqueList(ones)

    x0 = [int(point[0]) for point in zeros]
    y0 = [int(point[1]) for point in zeros]
    x1 = [int(point[0]) for point in ones]
    y1 = [int(point[1]) for point in ones]

    fig, ax = plt.subplots()
    fig.suptitle(idx)
    
    ax.set_xlabel("Attribute 0")
    ax.set_ylabel("Attribute 1")

    print(data[0])
    plt.xticks(np.arange(0, 5, 1.0))
    plt.yticks(np.arange(0, 5, 1.0))

    na, nb = (10, 10)  
    a = np.linspace(0, 4, na)  
    b = np.linspace(0, 4, nb)  
    xa, xb = np.meshgrid(a, b) 

    a = [0, 1, 2, 3, 4]
    b = [0, 1, 2, 3, 4]
    z = np.random.rand(5, 5)
    for indexa, ai in enumerate(a):
        for indexb, bi in enumerate(b):
            z[indexb][indexa] = driver.predict(tree, [ai, bi])
        
        #z.append(y)
    #z = z[:-1, :-1]
    #ax.fill_between() 

    ax.pcolormesh(a, b, z)

    ax.scatter(x0, y0, c=[[1,1,1]], marker='$0$')
    ax.scatter(x1, y1, c=[[0,0,0]], marker='$1$')

    plt.show()


#fig, ax = plt.subplots()

#doesn't print duplicates of points. makes the graph cleaner
#zeroes = driver.uniqueList(zeroes)
#ones = driver.uniqueList(ones)

#xZeroes = driver.colToList(zeroes, 0)
#yZeroes = driver.colToList(zeroes, 1)
#xOnes = driver.colToList(ones, 0)
#yOnes = driver.colToList(ones, 1)

#ax.scatter(xZeroes, yZeroes, c=[[0,0,0]], marker='$0$')
#ax.scatter(xOnes, yOnes, c=[[0,0,0]], marker='$1$')
#ax.set_xlabel("Attribute 0")
#ax.set_ylabel("Attribute 1")

#plt.xticks(np.arange(min(data[0])-1, max(data[0])+1, 1.0))
#plt.yticks(np.arange(min(data[1])-1, max(data[1])+1, 1.0))

#plt.show()




#for tree/plot 1:
#split on attribute zero first, that's a series of vertical lines at 0, 1, 2, 3, and 4











visualize(datas[0], trees[0], files[0])
visualize(datas[1], trees[1], files[1])
visualize(datas[2], trees[2], files[2])
visualize(datas[3], trees[3], files[3])