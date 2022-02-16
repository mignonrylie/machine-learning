import csv

class example:
    def __init__(self, f1, f2, label):
        self.f1 = f1
        self.f2 = f2
        self.label = label

points = []

def importData(filename): 
    csvData = [[]]
    with open(filename, newline = '') as file:
        reader = csv.reader(file, delimiter = ',')
        
        for row in reader:
            if row == []:
                continue
            
            dataPoint = []

            dataPoint.append(row[0])
            dataPoint.append(row[1])
            dataPoint.append(row[2])

            csvData.append(dataPoint)

    del csvData[0]
    return csvData