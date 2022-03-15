# -*- coding: utf-8 -*-

"""
Created on Mar. 11, 2019.
tensorflow implementation of the paper:
Dong-Kyu Chae et al. "CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks," In Proc. of ACM CIKM, 2018.
@author: Dong-Kyu Chae (kyu899@agape.hanyang.ac.kr)

"""

import random
from collections import defaultdict

def getHyperParams (benchmark):
    hyperParams = {}

    if benchmark == "Ciao":
        hyperParams['epochs'] = random.choice([450, 475, 500, 525, 550]) # quite sensitive to this value, hence trying multiple values
        
        hyperParams['mode'] = 'itemBased'
        hyperParams['hiddenDim_G'] = 250
        hyperParams['hiddenDim_D'] = 50
        hyperParams['reg_G'] = 0.001
        hyperParams['reg_D'] = 0.001
        hyperParams['lr_G'] = 0.0001
        hyperParams['lr_D'] = 0.0001
        hyperParams['batchSize_G'] = 128
        hyperParams['batchSize_D'] = 128
        hyperParams['opt_G'] = 'adam'
        hyperParams['opt_D'] = 'adam'
        hyperParams['hiddenLayer_G'] = 1
        hyperParams['hiddenLayer_D'] = 4
        hyperParams['step_G'] = 1
        hyperParams['step_D'] = 1
        
        # ZR => using zero-reconstruction 
        # PM => using partial-masking
        # ZP => using both 
        hyperParams['scheme'] = 'ZR'
        hyperParams['ZR_ratio'] = 40
        hyperParams['ZP_ratio'] = 20
        
        if hyperParams['scheme'] == 'ZP' or hyperParams['scheme'] == 'ZR':
            hyperParams['ZR_coefficient'] = 0.1
        else:
            hyperParams['ZR_coefficient'] = "None"
            hyperParams['ZR_ratio'] = "None"
            
        if hyperParams['scheme'] == 'ZR':
            hyperParams['ZP_ratio'] = "None"
    
    elif benchmark == "ML100K":
        hyperParams['epochs'] = random.choice([750, 800, 850, 900, 950, 1000]) # quite sensitive to this value, hence trying multiple values
        
        hyperParams['mode'] = 'itemBased'
        hyperParams['hiddenDim_G'] = 400
        hyperParams['hiddenDim_D'] = 125
        hyperParams['reg_G'] = 0.001
        hyperParams['reg_D'] = 0
        hyperParams['lr_G'] = 0.0001
        hyperParams['lr_D'] = 0.0001
        hyperParams['batchSize_G'] = 32
        hyperParams['batchSize_D'] = 64
        hyperParams['opt_G'] = 'adam'
        hyperParams['opt_D'] = 'adam'
        hyperParams['hiddenLayer_G'] = 1
        hyperParams['hiddenLayer_D'] = 1
        hyperParams['step_G'] = 4
        hyperParams['step_D'] = 2
        
        hyperParams['scheme'] = 'ZP'
        hyperParams['ZR_ratio'] = 70
        hyperParams['ZP_ratio'] = 70
        
        if hyperParams['scheme'] == 'ZP' or hyperParams['scheme'] == 'ZR':
            hyperParams['ZR_coefficient'] = 0.03
        else:
            hyperParams['ZR_coefficient'] = "None"
            hyperParams['ZR_ratio'] = "None"
            
        if hyperParams['scheme'] == 'ZR':
            hyperParams['ZP_ratio'] = "None"
    
    elif benchmark == "ML1M":
        hyperParams['epochs'] = random.choice([1475, 1500, 1525, 1550]) # quite sensitive to this value, hence trying multiple values

        hyperParams['mode'] = 'itemBased'
        hyperParams['hiddenDim_G'] = 300
        hyperParams['hiddenDim_D'] = 250
        hyperParams['reg_G'] = 0.001
        hyperParams['reg_D'] = 0.00005
        hyperParams['lr_G'] = 0.0001
        hyperParams['lr_D'] = 0.00005
        hyperParams['batchSize_G'] = 256
        hyperParams['batchSize_D'] = 512
        hyperParams['opt_G'] = 'adam'
        hyperParams['opt_D'] = 'adam'
        hyperParams['hiddenLayer_G'] = 1
        hyperParams['hiddenLayer_D'] = 1
        hyperParams['step_G'] = 1
        hyperParams['step_D'] = 1

        hyperParams['scheme'] = 'ZP'
        hyperParams['ZR_ratio'] = 90
        hyperParams['ZP_ratio'] = 90
        
        if hyperParams['scheme'] == 'ZP' or hyperParams['scheme'] == 'ZR':
            hyperParams['ZR_coefficient'] = 0.03
        else:
            hyperParams['ZR_coefficient'] = "None"
            hyperParams['ZR_ratio'] = "None"
            
        if hyperParams['scheme'] == 'ZR':
            hyperParams['ZP_ratio'] = "None"
            
    return hyperParams

    
