# -*- coding: utf-8 -*-

"""
Created on Mar. 11, 2019.
tensorflow implementation of the paper:
Dong-Kyu Chae et al. "CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks," In Proc. of ACM CIKM, 2018.
@author: Dong-Kyu Chae (kyu899@agape.hanyang.ac.kr)

"""

import tensorflow as tf
import numpy as np
import random
import math

tf.set_random_seed(123456789)
np.random.seed(123456789)
random.seed(123456789)

epsilon = 1e-3


def FullyConnectedLayer (input, inputDim, outputDim, activation, model, layer, reuse=False):
    scale1 = math.sqrt( 6 / (inputDim + outputDim) )
    
    wName = model+"_W"+str(layer)
    bName = model+"_B"+str(layer)
    
    with tf.variable_scope(model) as scope:
        
        if reuse == True:
            scope.reuse_variables()
        
        W = tf.get_variable(wName, [inputDim, outputDim], initializer=tf.random_uniform_initializer(-scale1, scale1))
        b = tf.get_variable(bName, [outputDim], initializer=tf.random_uniform_initializer(-0.01, 0.01))
    
        y = tf.matmul(input, W) + b
            
        L2norm = tf.nn.l2_loss(W)+tf.nn.l2_loss(b)
            
        if activation == "none":
            y = tf.identity(y, name="output")
            return y, L2norm, W, b
        
        elif activation == "sigmoid":
            return tf.nn.sigmoid(y), L2norm, W, b
        
        elif activation == "tanh":
            return tf.nn.tanh(y), L2norm, W, b

def GEN (input, userCount, h, activation, hiddenLayers):
    ZR_dims = tf.placeholder(tf.float32, [None, userCount])  
    
    #input->hidden                                                              
    y, L2norm, W, b = FullyConnectedLayer(input, userCount, h, activation, "gen", 0)  
    
    # stacked hidden layers
    for layer in range(hiddenLayers - 1):
        y, this_L2, W, b = FullyConnectedLayer(y, h, h, activation, "gen", layer+1)
        L2norm = L2norm + this_L2
    
    # hidden -> output
    y, this_L2, W, b = FullyConnectedLayer(y, h, userCount, "none", "gen", hiddenLayers+1)
    L2norm = L2norm + this_L2
    
    # loss function for ZR
    ZR_loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - 0)*ZR_dims, 1, keep_dims=True))
    
    return y, L2norm, ZR_loss, ZR_dims, 


def DIS(input, inputDim, h, activation, hiddenLayers, _reuse=False):
    #input->hidden                                                              
    y, _, W, b = FullyConnectedLayer(input, inputDim, h, activation, "dis", 0, reuse=_reuse)   
    
    # stacked hidden layers
    for layer in range(hiddenLayers - 1):
        y, _, W, b = FullyConnectedLayer(y, h, h, activation, "dis", layer+1, reuse=_reuse)
    
    # hidden -> output
    y, _, W, b = FullyConnectedLayer(y, h, 1, "none", "dis", hiddenLayers+1, reuse=_reuse)
    
    return y