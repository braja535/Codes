import csv
import pandas as pd
import io
import random
import math

#df = pd.read_csv('compute-03.csv',usecols=['time','value'])
#df = pd.read_csv('/home/raja/aggregation-cpu-average.percent-active_bb-blr-prod-compute.csv',usecols=['time','value'])
#df.to_csv('prod-compute.csv')


def LoadDatasets(filename,split,trainingSet=[],testset=[]):
    with open(filename,'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(2):
                dataset[x][y] = dataset[x][y]
            if random.random() < split:
                 trainingSet.append(dataset[x])
            else:
                testset.append(dataset[x])

def eucledian_distance(instance1,instance2,length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x]-instance2[x]),2)
    return math.sqrt(distance)


if __name__ == '__main__':

    train =[]
    test = []
    LoadDatasets('prod.csv',0.7,train, test)
    print 'Training set '  + repr(len(train))
    print 'Training set ' + repr(len(test))
    #for value in train:
    #    print value
    #print '****'
    #for value in test:
    #    print value

#with open('compute-03.csv') as csvfile:
#    lines = csv.reader(csvfile)

#    next(lines,None)
#    writer = csv.writer('compute-03_out.csv')
#    for row in lines:
#        print ','.join(row)
    #writer.close()
    #lines.close()