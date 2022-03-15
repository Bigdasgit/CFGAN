"""
Created on Mar. 11, 2019.
tensorflow implementation of the paper:
Dong-Kyu Chae et al. "CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks," In Proc. of ACM CIKM, 2018.
@author: Dong-Kyu Chae (kyu899@agape.hanyang.ac.kr)

usage: python cfgan.py
environment: python3.5xx, tensorflow_gpu
IMPORTANT: make sure that (1) the user & item indices start from 0, and (2) the index should be continuous, without any empy index.
"""

#cd Dropbox && cd Projects && cd CFGAN_for_open && python cfgan.py

import numpy as np
import trainer
import data 
import parameters 


######## prepare data ########
# IMPORTANT: make sure that (1) the user & item indices start from 0, and (2) there is NO EMPTY index (i.e., users or items having no ratings should be removed).

path = "datasets/"
benchmark = "Ciao"        # "Ciao", "ML100K", "ML1M"

######## load hyper-parameters ########
hyperParams = parameters.getHyperParams (benchmark)

mode = hyperParams['mode']        # 'userBased', 'itemBased'

# load purchase matrix on memory
trainSet, userCount, itemCount = data.loadTrainingData(benchmark, path)
testSet, GroundTruth = data.loadTestData(benchmark, path)
userList_test = list(testSet.keys())

# load all item (or user) purchase vectors on memory (thus, fast but consuming more memory) 
trainVector, testMaskVector, batchCount = data.to_Vectors (trainSet, userCount, itemCount, userList_test, mode)

# since we deal with implicit feedback
maskingVector = trainVector

# prepare for the random negative sampling (it is also memory-inefficient)
unobserved = []
for batchId in range(batchCount):
    unobserved.append(list( np.where(trainVector[batchId] == 0)[0] ) )


######## train CFGAN #########
topN = [5, 20]
useGPU = True
precision, recall, ndcg, mrr = trainer.trainCFGAN(userCount, itemCount, batchCount, trainVector, maskingVector, testMaskVector, userList_test, topN, unobserved, GroundTruth, hyperParams, mode, useGPU)

######## visualize learning curves (optional) #########
plotPath = "plots/"+benchmark+"/"
trainer.visualize(plotPath, precision, 'precision')
trainer.visualize(plotPath, recall, 'recall')
trainer.visualize(plotPath, ndcg, 'ndcg')
trainer.visualize(plotPath, mrr, 'mrr')
