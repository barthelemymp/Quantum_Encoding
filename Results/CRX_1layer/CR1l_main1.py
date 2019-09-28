# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:09:31 2019

@author: barthelemy
"""

#import libraries
import numpy as num
import pennylane as qml
import pennylane.templates.embeddings as embedding
import torch
import torch.autograd
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
# from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
# from sklearn.decomposition import PCA, KernelPCA
# from sklearn.model_selection import train_test_split
import math

from torchvision import transforms, utils



from qml_pyclass import * 
from qml_utils import *
from qml_dataload import * 


# Iris

csv_file = './iris-species/Iris.csv'
prep = Iris_binary_prep()
full_dataset = GenericDataset(csv_file, transform=None, preprocessing = prep)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

full_dataset.get_nfeatures()

for b in range(1,4):
    dev = qml.device('default.qubit', wires=2)
    CE_model = CE_1l(4,4)
    model = AEQC_CRX(dev, 2, CE_model, 4, b)
    file_name = 'CRX1l_Iris_AEQC_w'+str(2)+'_b'+str(b)+'_model_id'
    print(file_name)
    n_epoch = 30
    postprocessing = nn.LogSoftmax()
    loss_fn = torch.nn.CrossEntropyLoss()
    train_losses, valid_losses, valid_acc = train(model, train_dataset, test_dataset, loss_fn, n_epoch)
    train_losses = num.array(train_losses)
    num.save('./CRX1l_results/'+file_name+'train_losses',train_losses)
    valid_losses = num.array(valid_losses)
    num.save('./CRX1l_results/'+file_name+'valid_losses',valid_losses)
    valid_acc = num.array(valid_acc)
    num.save('./CRX1l_results/'+file_name+'valid_acc',valid_acc)
    torch.save(model, './CRX1l_results/model'+file_name)
    print('saved')
        
        

for w in range(2,6):
    for b in range(1,4):
        dev = qml.device('default.qubit', wires=w)
        CE_model = CE_1l(4,4)
        model = HQC_CRX(dev, w, CE_model, 4, b) #device, n_wires, CE, n_qfeatures, n_QClayers
        #TRAIN
        n_epoch = 30
        postprocessing = nn.LogSoftmax()
        loss_fn = torch.nn.CrossEntropyLoss()
        file_name = 'CRX1l_Iris_w'+str(w)+'_b'+str(b)+'_model_id'
        print(file_name)
        train_losses, valid_losses, valid_acc = train(model, train_dataset, test_dataset, loss_fn, n_epoch)
        train_losses = num.array(train_losses)
        num.save('./CRX1l_results/'+file_name+'train_losses',train_losses)
        valid_losses = num.array(valid_losses)
        num.save('./CRX1l_results/'+file_name+'valid_losses',valid_losses)
        valid_acc = num.array(valid_acc)
        num.save('./CRX1l_results/'+file_name+'valid_acc',valid_acc)
        torch.save(model, './CRX1l_results/model'+file_name)
        print('saved')
                
