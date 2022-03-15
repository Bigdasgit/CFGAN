# -*- coding: utf-8 -*-

"""
Created on Mar. 11, 2019.
tensorflow implementation of the paper:
Dong-Kyu Chae et al. "CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks," In Proc. of ACM CIKM, 2018.
@author: Dong-Kyu Chae (kyu899@agape.hanyang.ac.kr)

"""

import math

def computeTopNAccuracy (GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
        
        precision.append(sumForPrecision / len(predictedIndices))
        recall.append(sumForRecall / len(predictedIndices))
        NDCG.append(sumForNdcg / len(predictedIndices))
        MRR.append(sumForMRR / len(predictedIndices))
        
    return precision, recall, NDCG, MRR





