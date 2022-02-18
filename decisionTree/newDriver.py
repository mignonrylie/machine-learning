import onemore as driver

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

data, title = driver.importData("pokemonStats-discrete-TEST.csv")
del data[0]
datas.append(data)
root = driver.id3(data, -1, [x for x in range(len(data[0])-1)], titles=title)
trees.append(root)


#k-fold validation
#for each fold, generate a tree and 