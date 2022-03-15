# -*- coding: utf-8 -*-

"""
Created on Mar. 11, 2019.
tensorflow implementation of the paper:
Dong-Kyu Chae et al. "CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks," In Proc. of ACM CIKM, 2018.
@author: Dong-Kyu Chae (kyu899@agape.hanyang.ac.kr)

IMPORTANT: make sure that (1) the user & item indices start from 0, and (2) the index should be continuous, without any empy index.
"""

import random
import operator
import numpy as np
import codecs
from collections import defaultdict
from operator import itemgetter
import collections

def loadTrainingData (benchmark, path):    
    trainFile = path+"/"+benchmark+"/"+benchmark+".train"
    print(trainFile)
    
    trainSet = defaultdict(list)
    max_u_id = -1
    max_i_id = -1
    
    for line in open(trainFile):
        userId, itemId, rating = line.strip().split(' ')
        
        userId = int(userId)
        itemId = int(itemId)
        
        # note that we regard all the observed ratings as implicit feedback
        trainSet[userId].append(itemId)
                
        max_u_id = max(userId, max_u_id)
        max_i_id = max(itemId, max_i_id)
    
    for u, i_list in trainSet.items():
        i_list.sort()
        
    userCount = max_u_id+1
    itemCount = max_i_id+1
    
    print(userCount)
    print(itemCount)
    
    print("Training data loading done: %d users, %d items" % (userCount, itemCount))

    return trainSet, userCount, itemCount


def loadTestData (benchmark, path):
    testFile = path+"/"+benchmark+"/"+benchmark+".test"
    print(testFile)
    
    testSet = defaultdict(list)
    for line in open(testFile):
        userId, itemId, rating = line.strip().split(' ')
        userId = int(userId)
        itemId = int(itemId)
        
        # note that we regard all the ratings in the test set as ground truth
        testSet[userId].append(itemId)
        
    GroundTruth = []
    for u, i_list in testSet.items():
        tmp = []
        for j in i_list:
            tmp.append(j)

        GroundTruth.append(tmp)
        
    print("Test data loading done")
    
    return testSet, GroundTruth

    
def to_Vectors (trainSet, userCount, itemCount, userList_test, mode):
    # assume that the default is itemBased
    
    testMaskDict = defaultdict(lambda: [0] * itemCount)
    batchCount = itemCount
    if mode == "userBased":
        itemCount = userCount
        userCount = batchCount
        batchCount = itemCount
        
    trainDict = defaultdict(lambda: [0] * userCount)
   
    for userId, i_list in trainSet.items():
        for itemId in i_list:
            testMaskDict[userId][itemId] = -99999
            
            if mode == "userBased":
                trainDict[userId][itemId] = 1.0
            else:
                trainDict[itemId][userId] = 1.0
    
    trainVector = []
    for batchId in range(batchCount):
        trainVector.append(trainDict[batchId])
    
    testMaskVector = []
    for userId in userList_test:
        testMaskVector.append(testMaskDict[userId])

    print("Converting to vectors done....")
    
    return np.array(trainVector), np.array(testMaskVector), batchCount
    
    
    
    
