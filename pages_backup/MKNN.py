import pandas as pd
import numpy as np
import math


def euclidian_distance(train, test):
    distan = []
    for w in range(train.shape[0]):
        distance_index = []
        for i in range(test.shape[0]):
            sum = 0
            for j in range(test.shape[1]):
                uecli = ((train.iloc[w][j] - test.ilco[i][j]) ** 2)
                sum = sum + uecli
            distance_index.append(math.sqrt(sum))
        distan.append(distance_index)
    return distan

def validity(dataset):
    validity = []
    for w in range(dataset.shape[0]):
        eq = 0
        for i in range(dataset.shape[0] - 1):
            if dataset.iloc[w] == dataset.iloc[(w+i)%dataset.shape[0]]:
                eq += 1
        validity.append(eq/dataset.shape[0])
    return validity

def weight_voting(validity, euclidian):
    weight_voting = []
    validity = np.array(validity)
    euclidian = np.array(euclidian)
    for w in range(euclidian[i].shape[0]):
        weight_index = []
        for i in range(euclidian[i].shape[0]):
            weight = validity[w]*(1/(euclidian[w][i]+0.5))
            weight_index.append(weight)
        weight_voting.append(weight_index)
    return weight

def Classification_MKNN():
    pass