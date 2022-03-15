"""
Created on Mar. 11, 2019.
tensorflow implementation of the paper:
Dong-Kyu Chae et al. "CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks," In Proc. of ACM CIKM, 2018.
@author: Dong-Kyu Chae (kyu899@agape.hanyang.ac.kr)

"""

import tensorflow as tf
import numpy as np
import random
import time
import copy 

import evaluation 
import models 

def trainCFGAN(userCount, itemCount, batchCount, trainVector, maskingVector, testMaskVector, userList_test, topN, unobserved, GroundTruth, hp, mode, useGPU):

    # We consider <mode == itemBased> as default setting
    userCount_backUp = userCount
    itemCount_backUp = itemCount
    
    if mode == "userBased":
        itemCount = userCount
        userCount = itemCount_backUp
    
    # batch indicates a list of users if mode == userBased, and a list of items if mode == itemBased
    batchList = np.arange(batchCount)
    
    list_precision = []
    list_recall = []
    list_ndcg = []
    list_mrr = []
    
    with tf.Graph().as_default():
        # G
        # note that we do not use the random noise z.
        condition = tf.placeholder(tf.float32, [None, userCount])
        G_output, G_L2norm, G_ZR_loss, G_ZR_dims = models.GEN(condition, userCount, hp['hiddenDim_G'], 'sigmoid', hp['hiddenLayer_G'])
        
        
        # D
        mask = tf.placeholder(tf.float32, [None, userCount])        # purchased = 1, otherwise 0
        fakeData = G_output*mask
        fakeData = tf.concat([condition, fakeData], 1)
        
        realData = tf.placeholder(tf.float32, [None, userCount])
        _realData = tf.concat([condition, realData], 1)

        D_real = models.DIS(_realData, userCount*2, hp['hiddenDim_D'], 'sigmoid', hp['hiddenLayer_D'])
        D_fake = models.DIS(fakeData, userCount*2, hp['hiddenDim_D'], 'sigmoid', hp['hiddenLayer_D'], _reuse = True) 
        
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis')
        
        
        # define loss & optimizer for G
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake, labels = tf.ones_like(D_fake)))
        g_loss = g_loss + hp['reg_G']*G_L2norm
        if hp['scheme'] == 'ZP' or hp['scheme'] == 'ZR':
            g_loss = g_loss + hp['ZR_coefficient']*G_ZR_loss 
        
        if hp['opt_G'] == 'sgd':
            trainer_G = tf.train.GradientDescentOptimizer(hp['lr_G']).minimize(g_loss, var_list=g_vars)
        elif hp['opt_G'] == 'adam':
            trainer_G = tf.train.AdamOptimizer(hp['lr_G']).minimize(g_loss, var_list=g_vars)
            
        
        # define loss & optimizer for D
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real, labels = tf.ones_like(D_real))) 
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake, labels = tf.zeros_like(D_fake))) 
        
        D_L2norm = 0
        for pr in d_vars:
            D_L2norm = D_L2norm + tf.nn.l2_loss(pr)
        d_loss = d_loss_real + d_loss_fake + hp['reg_D']*D_L2norm
        
        if hp['opt_D'] == 'sgd':
            trainer_D = tf.train.GradientDescentOptimizer(hp['lr_D']).minimize(d_loss, var_list=d_vars)
        elif hp['opt_D'] == 'adam':
            trainer_D = tf.train.AdamOptimizer(hp['lr_D']).minimize(d_loss, var_list=d_vars)
        
        # define top-K
        prediction_top_k = tf.placeholder(tf.float32, [None, None])
        scale_top_k = tf.placeholder(tf.int32)
        top_k = tf.nn.top_k(prediction_top_k, scale_top_k)        

        
        # start training
        if useGPU == True:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
        else:
            config = tf.ConfigProto(device_count = {'GPU': 0})
        
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        
        totalEpochs = hp['epochs']
        totalEpochs = int(totalEpochs / hp['step_G'])
        
        for epoch in range(totalEpochs):
            # we first perform the negative sampling for PM and ZR for this epoch
            n_samples_ZR = []
            n_samples_PM = []
            
            for batchId in range(batchCount):
                # ZR
                if hp['scheme'] == 'ZP' or hp['scheme'] == 'ZR':
                    seed = int(time.time())
                    np.random.seed(seed)
                    n_samples_ZR.append(np.random.choice(unobserved[batchId], int(len(unobserved[batchId])*hp['ZR_ratio']/100), replace=False))
                
                # PM
                if hp['scheme'] == 'ZP' or hp['scheme'] == 'PM':
                    seed = int(time.time())
                    np.random.seed(seed)
                    n_samples_PM.append(np.random.choice(unobserved[batchId], int(len(unobserved[batchId])*hp['ZP_ratio']/100), replace=False))
                
            # D step
            numOfMinibatches = int(len(batchList) / hp['batchSize_D']) + 1
            numOfLastMinibatch = len(batchList) % hp['batchSize_D']
            
            for epoch_D in range(hp['step_D']):
                t1 = time.time()
                loss_d = 0
                
                random.seed(seed)
                random.shuffle(batchList)

                for batchId in range(numOfMinibatches):
                    start = batchId * hp['batchSize_D']
                    if batchId == numOfMinibatches-1:   #if it is the last minibatch
                        numOfBatches = numOfLastMinibatch
                    else:
                        numOfBatches = hp['batchSize_D']
                    end = start + numOfBatches
                    batchIndex = batchList[start: end]
                    
                    trainMask = []
                    for batchId in batchIndex:
                        # PM, convert 0 to 1 in the masking vector e_u
                        if hp['scheme'] == 'ZP' or hp['scheme'] == 'PM':
                            tmp = np.copy(maskingVector[batchId])
                            tmp[n_samples_PM[batchId]] = 1
                            trainMask.append(tmp)
                        else:
                            trainMask.append(maskingVector[batchId])
                            
                    _, dLoss = sess.run([trainer_D, d_loss], feed_dict={realData: trainVector[batchIndex], mask: trainMask, condition: trainVector[batchIndex]}) #Update D
                    loss_d = loss_d + dLoss

                t2 = time.time()
                print("[%d: D][%d] cost:%.4f, within %s seconds" % (epoch + 1, epoch_D + 1, loss_d, str(t2-t1)))
            
            # G step
            numOfMinibatches = int(len(batchList) / hp['batchSize_G']) + 1
            numOfLastMinibatch = len(batchList) % hp['batchSize_G']
            for epoch_G in range(hp['step_G']):
                t1 = time.time()
                loss_g = 0
                
                random.seed(seed)
                random.shuffle(batchList)
            
                for batchId in range(numOfMinibatches):
                    start = batchId * hp['batchSize_G']
                    if batchId == numOfMinibatches-1:   #if it is the last minibatch
                        numOfBatches = numOfLastMinibatch
                    else:
                        numOfBatches = hp['batchSize_G']
                    end = start + numOfBatches
                    batchIndex = batchList[start: end]
                    
                    trainMask = []
                    trainZRMask = []
                    
                    for batchId in batchIndex:
                        # PM, convert 0 to 1 in the masking vector e_u
                        if hp['scheme'] == 'ZP' or hp['scheme'] == 'PM':
                            tmp = np.copy(maskingVector[batchId])
                            tmp[n_samples_PM[batchId]] = 1
                            trainMask.append(tmp)
                        else:
                            trainMask.append(maskingVector[batchId])
                            
                        # ZR
                        tmp = np.zeros(userCount)
                        if hp['scheme'] == 'ZP' or hp['scheme'] == 'ZR':
                            tmp[n_samples_ZR[batchId]] = 1
                            trainZRMask.append(tmp)
                        else:
                            trainZRMask.append(tmp)
                            
                    _, gLoss = sess.run([trainer_G, g_loss], feed_dict={condition: trainVector[batchIndex], mask: trainMask, realData: trainVector[batchIndex], G_ZR_dims: trainZRMask}) #Update G 
                    loss_g = loss_g + gLoss
                    
                print("[%d: G][%d] cost : %.4f, within %s seconds" % (epoch + 1, epoch_G+1, loss_g, str(t2-t1)))
                
                
                # measure accuracy
                t1 = time.time()
                allRatings = sess.run(G_output, feed_dict={condition: trainVector})
                if mode == "itemBased":
                    allRatings = np.transpose(allRatings)
                allRatings = allRatings[userList_test] + testMaskVector
                
                _, predictedIndices = sess.run(top_k, feed_dict={prediction_top_k: allRatings, scale_top_k: topN[-1]})
                precision, recall, ndcg, mrr = evaluation.computeTopNAccuracy (GroundTruth, predictedIndices, topN)
                
                t2 = time.time()
                
                # print top-5 results for each epoch
                print("[G][%d-%d evaluation within %s seconds] precision: %.4f, recall: %.4f, ndcg: %.4f, mrr: %.4f" % (epoch + 1, epoch_G + 1, str(t2-t1), precision[0], recall[0], ndcg[0], mrr[0]))
                
                list_precision.append(precision[0])
                list_recall.append(recall[0])
                list_ndcg.append(ndcg[0])
                list_mrr.append(mrr[0])
                
        sess.close()
    
    print("completed")
    for index, N in enumerate(topN):
        print("top-%d accuracy, precision: %.4f, recall: %.4f, ndcg: %.4f, mrr: %.4f" % (N, precision[index], recall[index], ndcg[index], mrr[index]))
    
    # for visualization, if needed
    return list_precision, list_recall, list_ndcg, list_mrr

    
import matplotlib.pyplot as plt
def visualize(plotPath, values, whatMetric):
    plt.figure(figsize=(6,4))
    plt.rcParams['lines.linewidth'] = 3
    
    ylim = max(values)*1.2
    axes = plt.gca()
    axes.set_ylim([0, ylim])

    x = range(1, len(values)+1)
    plt.plot(x, values, label = (whatMetric+',maxVal='+str(max(values))+',lastVal'+str(values[-1])))
    plt.grid()
    plt.legend()
    plt.savefig(plotPath+whatMetric+'.top5.png')
    