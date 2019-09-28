#import libraries
import pennylane as qml
import torch
import torch.autograd
import pennylane.templates.embeddings as embedding
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.model_selection import train_test_split
import math
from qml_utils import *



class HQC(nn.Module):
    def __init__(self,device, n_wires, CE, n_qfeatures, n_QClayers):
        super(HQC, self).__init__()
        self.device = device
        self.n_wires =n_wires
        self.mode = torch.tensor(1, dtype = torch.uint8)
        self.n_qfeatures = n_qfeatures #dimension of the output of CE
        self.n_QClayers = n_QClayers #numbers of layers in the Quantum classifier
        self.n_QElayers = math.ceil(self.n_qfeatures/n_wires) #number of layer necessary to encode my data
        print('qfeatures:',self.n_qfeatures,'divided by:',self.n_wires, 'self.n_QElayers', self.n_QElayers)
        self.to_pad = self.n_QElayers*n_wires - self.n_qfeatures #number of layer necessary to encode my data
        print('self.to_pad',self.to_pad)
        #assert self.to_pad == 0,"CE exit dim is not a multiple of n_wires"
        #CE
        self.CE = CE # before I was using nn.Sequential(*CE_layers), may still be a good idea
        #QE
        self.QE_optvar = torch.nn.Parameter(torch.randn((self.n_QElayers, 2, n_wires), requires_grad =True))
        #QC
        self.QC_optvar = torch.nn.Parameter(torch.randn((self.n_QClayers, 3, n_wires), requires_grad =True))
        
        
    def trainmode_(self,):
        self.mode = torch.tensor(1, dtype = torch.uint8)
        
    def evalmode_(self,):
        self.mode = torch.tensor(0, dtype = torch.uint8)
        
    def forward(self, x):

        #CE        
        #x = x.view(-1, self.num_flat_features(x)) #usefull when batched!
        x = self.CE(x)
        #QE 
        x =F.pad(x, (0,self.to_pad))
        x = x.view((self.n_QElayers, 1, self.n_wires)) #reshape so that it can be concat with the learnable parameters
        QE_param = torch.cat((x,self.QE_optvar,), dim = 1) # concat them to make them fit in stronglayer
        # if we are in training_mode
        if(self.mode):
            #QE 
            @qml.qnode(self.device, interface='torch')
            def circuit(x,QE_param,QC_optvar):
                #QE 
                for j in range(self.n_QElayers):
                    QE_layparam = QE_param[j,:,:]
                    stronglayer(QE_layparam,r=1, n_wires = self.n_wires)
                for i in range(self.n_QClayers):
                    QC_layparam = QC_optvar[i,:,:]
                    stronglayer(QC_layparam,r=1, n_wires = self.n_wires)
                return qml.expval.PauliZ(0)
            
        else:
            @qml.qnode(self.device, interface='torch')
            def circuit(x,QE_param,QC_optvar):
                #QE 
                for j in range(self.n_QElayers):
                    QE_layparam = QE_param[j,:,:]
                    stronglayer(QE_layparam,r=1, n_wires = self.n_wires)
                return qml.expval.PauliZ(0)

        #QC
        out = 0.5*(1+(circuit(x,QE_param,self.QC_optvar)))
        out = out.unsqueeze(0)

        return torch.cat((out,1-out,))

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        print('numflat',num_features)
        return num_features

class HQC_CRX(nn.Module):
    def __init__(self,device, n_wires, CE, n_qfeatures, n_QClayers):
        super(HQC_CRX, self).__init__()
        self.device = device
        self.n_wires =n_wires
        self.mode = torch.tensor(1, dtype = torch.uint8)
        self.n_qfeatures = n_qfeatures #dimension of the output of CE
        self.n_QClayers = n_QClayers #numbers of layers in the Quantum classifier
        self.n_QElayers = math.ceil(self.n_qfeatures/n_wires) #number of layer necessary to encode my data
        print('self.n_QElayers', self.n_QElayers)
        self.to_pad = self.n_QElayers*n_wires - self.n_qfeatures #number of layer necessary to encode my data
        print('self.to_pad',self.to_pad)
        #assert self.to_pad == 0,"CE exit dim is not a multiple of n_wires"
        #CE
        self.CE = CE # before I was using nn.Sequential(*CE_layers), may still be a good idea
        #QE
        self.QE_optvar = torch.nn.Parameter(torch.randn((self.n_QElayers, 2, n_wires), requires_grad =True))
        self.QE_CRX_optvar = torch.nn.Parameter(torch.randn((self.n_QElayers, n_wires), requires_grad =True))
        #QC
        self.QC_optvar = torch.nn.Parameter(torch.randn((self.n_QClayers, 3, n_wires), requires_grad =True))
        self.QC_CRX_optvar = torch.nn.Parameter(torch.randn((self.n_QClayers, n_wires), requires_grad =True))
        
        
    def trainmode_(self,):
        self.mode = torch.tensor(1, dtype = torch.uint8)
        
    def evalmode_(self,):
        self.mode = torch.tensor(0, dtype = torch.uint8)
        
    def forward(self, x):
        #CE        
        #x = x.view(-1, self.num_flat_features(x)) #usefull when batched!
        x = self.CE(x)
        #QE 
        x =F.pad(x, (0,self.to_pad))
        x = x.view((self.n_QElayers, 1, self.n_wires)) #reshape so that it can be concat with the learnable parameters
        QE_param = torch.cat((x,self.QE_optvar,), dim = 1) # concat them to make them fit in stronglayer
        # if we are in training_mode
        if(self.mode):
            #QE 
            @qml.qnode(self.device, interface='torch')
            def circuit(x,QE_param, QE_CRX_optvar, QC_optvar, QC_CRX_optvar):
                #QE 
                for j in range(self.n_QElayers):
                    QE_layparam = QE_param[j,:,:]
                    QE_CRX_layparam = QE_CRX_optvar[j,:]
                    stronglayerCRX(QE_layparam, QE_CRX_layparam, r=1, n_wires = self.n_wires)
                for i in range(self.n_QClayers):
                    QC_layparam = QC_optvar[i,:,:]
                    QC_CRX_layparam = QC_CRX_optvar[i,:]
                    stronglayerCRX(QC_layparam, QC_CRX_layparam, r=1, n_wires = self.n_wires)
                return qml.expval.PauliZ(0)
            
        else:
            @qml.qnode(self.device, interface='torch')
            def circuit(x,QE_param, QE_CRX_optvar, QC_optvar, QC_CRX_optvar):
                #QE 
                for j in range(self.n_QElayers):
                    QE_layparam = QE_param[j,:,:]
                    QE_CRX_layparam = QE_CRX_optvar[j,:]
                    stronglayerCRX(QE_layparam, QE_CRX_layparam, r=1, n_wires = self.n_wires)
                return qml.expval.PauliZ(0)

        #QC
        out = 0.5*(1+(circuit(x,QE_param,self.QE_CRX_optvar, self.QC_optvar, self.QC_CRX_optvar)))
        out = out.unsqueeze(0)

        return torch.cat((out,1-out,))

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        print('numflat',num_features)
        return num_features

class AEQC(nn.Module):
    def __init__(self,device, n_wires, CE, n_qfeatures, n_QClayers):
        super(AEQC, self).__init__()
        self.device = device
        self.n_wires =n_wires
        self.mode = torch.tensor(1, dtype = torch.uint8)
        self.n_qfeatures = n_qfeatures #dimension of the output of CE
        self.n_QClayers = n_QClayers
        self.to_pad = 2**n_wires - n_qfeatures
        #CE
        self.CE = CE #nn.Sequential(*CE_layers)
        #QE
        self.wires_to_encode = list(range(0,n_wires))
        #QC
        self.QC_optvar = torch.nn.Parameter(torch.randn((self.n_QClayers, 3, n_wires), requires_grad =True))
        
    def forward(self, x):
        #CE        
        #x = x.view(-1, self.num_flat_features(x)) #usefull when batched!
        x = self.CE(x)
        #QE 
        #x = torch.unsqueeze(x, dim=0)
        
        x =F.pad(x, (0,self.to_pad))

        @qml.qnode(self.device, interface='torch')
        def circuit(x,QC_optvar):
            #QE 
            embedding.AmplitudeEmbedding(x, self.wires_to_encode)
            #QCyu
            for i in range(self.n_QClayers):
                QC_layparam = QC_optvar[i,:,:]
                stronglayer(QC_layparam,r=1, n_wires = self.n_wires)
            return qml.expval.PauliZ(0)


        #QC
        out = 0.5*(1+(circuit(x,self.QC_optvar)))
        out = out.unsqueeze(0)
#         print(type(out),out,out.shape)
#         print('req grad dans le forward', out.requires_grad)
        return torch.cat((out,1-out,))#0.5*(1+(circuit(x,QE_param,self.QC_optvar)))

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        print('numflat',num_features)
        return num_features

        
    
    
class AEQC_CRX(nn.Module):
    def __init__(self,device, n_wires, CE, n_qfeatures, n_QClayers):
        super(AEQC_CRX, self).__init__()
        self.device = device
        self.n_wires =n_wires
        self.mode = torch.tensor(1, dtype = torch.uint8)
        self.n_qfeatures = n_qfeatures #dimension of the output of CE
        self.n_QClayers = n_QClayers
        self.to_pad = 2**n_wires - n_qfeatures
        #CE
        self.CE = CE #nn.Sequential(*CE_layers)
        #QE
        self.wires_to_encode = list(range(0,n_wires))
        #QC
        self.QC_optvar = torch.nn.Parameter(torch.randn((self.n_QClayers, 3, n_wires), requires_grad =True))
        self.QC_CRX_optvar = torch.nn.Parameter(torch.randn((self.n_QClayers, n_wires), requires_grad =True))
        
    def forward(self, x):
        #CE        
        #x = x.view(-1, self.num_flat_features(x)) #usefull when batched!
        x = self.CE(x)
        #QE 
        #x = torch.unsqueeze(x, dim=0)
        
        x =F.pad(x, (0,self.to_pad))

        @qml.qnode(self.device, interface='torch')
        def circuit(x,QC_optvar,QC_CRX_optvar):
            #QE 
            embedding.AmplitudeEmbedding(x, self.wires_to_encode)
            #QC
            for i in range(self.n_QClayers):
                QC_layparam = QC_optvar[i,:,:]
                QC_CRX_layparam = QC_CRX_optvar[i,:]
                stronglayerCRX(QC_layparam, QC_CRX_layparam, r=1, n_wires = self.n_wires)
            return qml.expval.PauliZ(0)


        #QC
        out = 0.5*(1+(circuit(x,self.QC_optvar, self.QC_CRX_optvar)))
        out = out.unsqueeze(0)
        return torch.cat((out,1-out,))#0.5*(1+(circuit(x,QE_param,self.QC_optvar)))

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        print('numflat',num_features)
        return num_features



def train(model,train_dataset,test_dataset,loss_fn,n_epoch=5):
    #model.trainmode_()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    valid_losses = []
    valid_acc = []
    train_losses = []
    train_loss_store = torch.zeros(1)
    for e in range(n_epoch):
        print('n_epoch',e)
        for i in range(len(train_dataset)):
            #print('train_data i',i)
            
            #eval
            if (i == 0):
                score = torch.zeros(1)
                valid_loss_store = torch.zeros(1)
                
                with torch.no_grad():
                    for j in range(len(test_dataset)):
                        #print('evalmode', j)
                        sample = test_dataset[j]
                        X, label = sample['X'], sample['label']
                        output = model(X)
                        output = torch.unsqueeze(output,0)
                        label = torch.unsqueeze(label,0)

                        loss = loss_fn(output, label)
                        valid_loss_store += loss
                        
                        _, predicted = torch.max(output.data, 1)
                        #print('predicted', predicted.data, 'label', label.data)
                        score += predicted.eq(label).float()
                score = score/len(test_dataset)
                print('acc', score)
                valid_acc += [score.item()]
                valid_loss_store = valid_loss_store/len(test_dataset)
                print('valid_loss_store',valid_loss_store)
                valid_losses += [valid_loss_store.item()]
                train_loss_store = train_loss_store/20
                print('train_loss_store',train_loss_store)
                train_losses += [train_loss_store.item()]
                train_loss_store = torch.zeros(1)
                    


                    
            sample = train_dataset[i]
            X, label = sample['X'], sample['label']
            #print(label)

            optimizer.zero_grad()
            output = model(X)

            
            
            #artificially adding the batch dimention to make it compatible with the loss function
            output = torch.unsqueeze(output,0)
            label = torch.unsqueeze(label,0)
            #output = torch.log(output)#postprocessing(output)
            
            loss = loss_fn(output, label)
            #loss = - label*torch.log(output) - (1-label)*torch.log(1-output)
            train_loss_store += loss
            
            #print('loss', loss)
            loss.backward()
#             for name, param in model.named_parameters():
#                 #print(param)
#                 print(i,name,'gradient odg',param.grad.data.sum())
            optimizer.step()
    return train_losses, valid_losses, valid_acc
            
            
