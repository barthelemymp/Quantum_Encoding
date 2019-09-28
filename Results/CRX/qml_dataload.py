# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:42:15 2019

@author: barthelemy
"""

import pennylane as qml
import torch
import torch.autograd
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.model_selection import train_test_split
import math
from qml_utils import *






class GenericDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None, preprocessing =None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            preprocessing (callable, optional): Optional preprocessing to be applied
                on the complete dataset. can be label, pca, etc
        """
        
        #load the data
        self.data = pd.read_csv(csv_file)
        
        #apply the preprocessing, to make easier to work with
        if preprocessing :
            self.data = preprocessing(self.data)
        
        # shuffle the data
        self.data= self.data.sample(frac=1)
        
        #separate features from labels
        self.Xs = self.data.iloc[:,0:-1]
        self.labels = self.data.iloc[:,-1]
        
        #other attributes
        self.transform = transform
        self.nfeatures = len(self.Xs.iloc[0,:])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #print(type(idx))
        #convert the idx normal list, to make it compatible with random_split
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = torch.tensor(self.Xs.iloc[idx,:].as_matrix()).float() #take the features, locate the idx, to numpy, to tensor, to floattensor
        Y = torch.tensor(self.labels.iloc[idx]).long() #take the features, locate the idx, to numpy, to tensor, to longtensor
                                                       #remark: the long is for the nn.CrossEntropy
        #put it into a dictionnary (not sure to know why, but it s the official way)
        sample = {'X': X, 'label': Y}
        #apply the transform to the selected element
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def get_nfeatures(self,): #give the number of features after preprocessing and transform.
        sample = self[0]
        X = sample['X']
        return(X.shape[0])
        
            
        
######## DataLoader transform
        
class Polynomial_transform(object):
    
#     """extend the features to their polynomial compinastion, up to a specified degree.

#     Args:
#         degree of the polynomial extension
#     """

    def __init__(self, degree):
        self.degree = degree
        self.poly = PolynomialFeatures(self.degree) #scikit learn polynomial extension

    def __call__(self, sample):
        X, Y = sample['X'], sample['label']
        xsize = X.shape
        X = X.unsqueeze(dim=0)
        X = torch.tensor(self.poly.fit_transform(X)).float()
        X = X.squeeze()
        sample = {'X': X, 'label': Y}
        return sample
    
class Yeo_Johnson_transform(object):
    
    def __init__(self, standardize = False):
        self.standardize = standardize
        self.pt = PowerTransformer(method='yeo-johnson', standardize=self.standardize) #scikit learn Powertransform
                                            # the method 'box-cox' could also work but only with positive values

    def __call__(self, sample):
        X, Y = sample['X'], sample['label']
        xsize = X.shape
        X = X.unsqueeze(dim=0) #make it batch like
        X = torch.tensor(self.poly.fit_transform(X)).float()
        X = X.squeeze() #back to the sample real dim
        sample = {'X': X, 'label': Y}
        return sample
    
######## Data preprocessing 
        
class Iris_binary_prep(object): #preprocessing for the Iris dataset
    def __init__(self,):
        super(Iris_binary_prep, self).__init__()
        
    def __call__(self, data): #data is in panda format
        data = data.replace('Iris-setosa',0)
        data = data.replace('Iris-virginica',1)
        data = data.replace('Iris-versicolor',2)
        data = data[data.iloc[:,-1] != 2]
        data = data.drop(columns="Id")
        data = pd.DataFrame(data.as_matrix(), columns=['SepalLengthCm'  ,'SepalWidthCm','PetalLengthCm','PetalWidthCm', 'Species'])
        return data
       
class BC_binary_prep(object): #preprocessing for the BC dataset
    def __init__(self,):
        super(BC_binary_prep, self).__init__()
        
    def __call__(self, data): #data is in panda format
        data = data.drop(['Unnamed: 32', 'id'], axis = 1)
        cols = data.columns.tolist()
        cols = cols[1:] + [cols[0]]
        data = data[cols]
#         data['diagnosis'].loc[data['diagnosis'] == 'M'] = 0
#         data['diagnosis'].loc[data['diagnosis'] == 'B'] = 1
        diag = { "M": 0, "B": 1}
        data["diagnosis"].replace(diag, inplace=True)
        return data

class Yeo_Johnson_prep(object): #Yeo Johnson transformation but applied to the whole dataset,
                                #enables to compute reasonable mean and std for the case of standardize==True
    def __init__(self, standardize = False):
        super(Yeo_Johnson_prep, self).__init__()
        self.standardize = standardize
        self.pt = PowerTransformer(method= 'yeo-johnson', standardize=self.standardize)#scikit learn Powertransform
                                            # the method 'yeo-johnson','box-cox' could also work but only with positive values

        
    def __call__(self, data): #data is in panda format, after the first preprocessing has been applied
        x = data.iloc[:,0:-1] 
        data.iloc[:,0:-1] = self.pt.fit_transform(x)
        return data
    
class PCA_prep(object): #apply a pca 
    
    def __init__(self, n_components):
        
        self.n_components = n_components
        self.pca = PCA(self.n_components)

    def __call__(self, data):
        x = data.iloc[:,0:-1]
        labels = data.iloc[:,-1]
        x = pd.DataFrame(self.pca.fit_transform(x)).copy()
        data = pd.concat([x,labels], axis=1).copy()
        return data
    
class Kernel_PCA_prep(object): #apply a kernel pca, default is gaussian
    
    def __init__(self, n_components, kernel = 'rbf'):
        self.kernel = kernel
        self.n_components = n_components
        self.kpca = KernelPCA(self.n_components)

    def __call__(self, data):
        x = data.iloc[:,0:-1]
        labels = data.iloc[:,-1]
        x = pd.DataFrame(self.kpca.fit_transform(x))
        data = pd.concat([x,labels], axis=1)
        return data
