# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:59:08 2019

@author: barthelemy
"""

#import libraries
import pennylane as qml
import torch
import torch.autograd
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
# from sklearn.preprocessing import PolynomialFeatures 
import math

CE_layers = [nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU()]
    
#making an highly entangling layer

def stronglayer(slparam, n_wires, r=1):
    slparam = math.pi * slparam

    qml.RX(slparam[0,0], wires = 0)
    qml.RY(slparam[1,0], wires = 0)
    qml.RZ(slparam[2,0], wires = 0)
    for i in range(1,n_wires):

        qml.RX(slparam[0,i], wires = i)
        qml.RY(slparam[1,i], wires = i)
        qml.RZ(slparam[2,i], wires = i)
        qml.CNOT(wires=[i, i-1])
    qml.CNOT(wires=[0, n_wires-1])
    
def stronglayerCRX(slparam, n_wires, r=1):
    slparam = math.pi * slparam

    qml.RX(slparam[0,0], wires = 0)
    qml.RY(slparam[1,0], wires = 0)
    qml.RZ(slparam[2,0], wires = 0)
    for i in range(1,n_wires):

        qml.RX(slparam[0,i], wires = i)
        qml.RY(slparam[1,i], wires = i)
        qml.RZ(slparam[2,i], wires = i)
        qml.CRX(phi =slparam[3,i],wires=[i, i-1])
    qml.CRX(phi =slparam[3,0], wires=[0, n_wires-1])
    
class CE_2l(torch.nn.Module):
    def __init__(self,):
        super(CE_2l, self).__init__()
        self.l1 = torch.nn.Linear(30,16)
        self.l2 = torch.nn.Linear(16,4)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        out1 = self.sigmoid(self.l1(x)) 
        #y_pred = self.sigmoid(self.l2(out1)) - 0.5
        y_pred = self.tanh(self.l2(out1))
        return y_pred
    
#One layer Calssical encoder
class CE_1l(torch.nn.Module):
    def __init__(self,out_dim = 4):
        super(CE_1l, self).__init__()
        self.out_dim = out_dim
        self.l1 = torch.nn.Linear(30,out_dim)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        #out1 = self.sigmoid(self.l1(x)) - 0.5
        out1 = self.tanh(self.l1(x)) - 0.5
        return out1
    
class CE_id(torch.nn.Module):
    def __init__(self):
        super(CE_id, self).__init__()
        #self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        #out1 = self.sigmoid(x) - 0.5
        out1 = self.tanh(x)
        return out1
    