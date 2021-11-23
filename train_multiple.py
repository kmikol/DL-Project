#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 12:30:39 2021

@author: s202818
"""



import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import copy
import time
from data_loader import Loader, splitData
import os
import pickle

IMG_PATH = "images/"
N_RUNS = 10
NUM_EPOCHS = 250

train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomAffine(degrees=(-180,180),
                                                             translate=(0.1,0.1),
                                                             scale=(0.9,1.1),
                                                             shear=25),
                                      transforms.ColorJitter(brightness=0.05,
                                                             contrast=0.05,
                                                             saturation=0.05,
                                                             hue=0.05)
                                      ])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'The model will run on: {device}')


class Model(torch.nn.Module):
    def __init__(self,num_diagnosis,num_characteristics,base_model = 'efficientnet'):
        super(Model,self).__init__()
        
        if base_model == 'densenet':
            model = models.densenet201(pretrained=True)
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_diagnosis+num_characteristics)
        
        elif base_model == 'resnet':
            model = models.resnet50(pretrained=True)
            model.fc = torch.nn.Linear(model.fc.in_features, num_diagnosis+num_characteristics)
            
        elif base_model == 'efficientnet':
            model = models.efficientnet_b2(pretrained=True)
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_diagnosis+num_characteristics)
        else:
            raise Exception('base model not available')
        
        for param in model.parameters():
            param.requires_grad=True
        
        self.model = model
        self.num_diagnosis = num_diagnosis
        self.num_characteristics = num_characteristics
        
    def forward(self,x):
        
        y = self.model(x)
        diagnosis_enc = y[:,:self.num_diagnosis]
        characteristic_enc = y[:,self.num_diagnosis:]
        
        return diagnosis_enc, characteristic_enc

      
def lossFunction(y_hat,y,diagnosis_weight,characteristic_weight):
    
    def bce(y_hat,y,weights = characteristic_weight ):
        p = torch.nn.Sigmoid()(torch.clip(y_hat,-10,10))
        loss = -torch.mean(characteristic_weight*y*torch.log(p) + (1-characteristic_weight)*(1-y)*torch.log(1-p))
        return loss
    
    y_hat_diagnosis = y_hat[0]
    y_hat_characteristic = y_hat[1]
    y_diagnosis = y[0]
    y_characteristic = y[1]
        
    
    xEntropy = torch.nn.CrossEntropyLoss(weight=diagnosis_weight.to(device),reduce='mean')
    #bce = torch.nn.BCELoss(reduce='mean')
    sigmoid = torch.nn.Sigmoid()
    xEntropy_loss = xEntropy(y_hat_diagnosis,y_diagnosis)
    bce_loss = bce(y_hat_characteristic,y_characteristic)
    
    if bce_loss<0:
        print(sigmoid(y_hat_characteristic))
        print(y_characteristic)
        print(bce_loss)
    
    return xEntropy_loss, bce_loss

def train(model,train_set,test_set,lossFunction,optimizer,diagnosis_weight,characteristics_weight,
          num_epochs=10,batch_size=64,num_workers=16, save_path=None):
    
    history = {'train_loss':[],
               'train_diagnosis_loss':[],
               'train_characteristic_loss':[],
               'train_accuracy':[],
               'test_loss':[],
               'test_diagnosis_loss':[],
               'test_characteristic_loss':[],
               'test_accuracy':[]}
    
    N_train = len(train_set)
    N_test = len(test_set)
    
    train_loader = DataLoader(train_set,batch_size=batch_size,
                              shuffle=True,num_workers=num_workers)
    
    test_loader = DataLoader(test_set,batch_size=batch_size,
                              shuffle=False,num_workers=num_workers)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    
    for epoch_idx in range(num_epochs):
        
        train_iterator = iter(train_loader)
        test_iterator = iter(test_loader)
                
        train_loss = 0
        train_diagnosis_loss = 0
        train_characteristic_loss = 0
        train_correct = 0
        
        test_loss = 0
        test_diagnosis_loss = 0
        test_characteristic_loss = 0
        test_correct = 0
        
        model.train()
        for img,disease,characteristic in tqdm(train_iterator, desc='train: '):
            X = img.to(device).float()
            y = [disease.to(device), characteristic.to(device)]
            
            curr_batch_size = disease.shape[0]
            
            optimizer.zero_grad()
            y_hat = model(X)
            xEntropy_loss, bce_loss = lossFunction(y_hat,y,diagnosis_weight,characteristics_weight)
            loss = xEntropy_loss + bce_loss
            loss.backward()
            optimizer.step()
            
            train_loss += curr_batch_size * loss / N_train
            train_diagnosis_loss += curr_batch_size * xEntropy_loss / N_train
            train_characteristic_loss += curr_batch_size * bce_loss / N_train
            
            predicted = y_hat[0].argmax(1)
            train_correct += (y[0]==predicted).sum().item()
        
        model.eval()
        for img,disease,characteristic in tqdm(test_iterator, desc='test: '):
            
            with torch.no_grad():
                X = img.to(device).float()
                y = [disease.to(device), characteristic.to(device)]
                
                curr_batch_size = disease.shape[0]
            
                y_hat = model(X)
                xEntropy_loss, bce_loss = lossFunction(y_hat,y,diagnosis_weight,characteristics_weight)
                loss = xEntropy_loss + bce_loss
            
                test_loss += curr_batch_size * loss / N_test
                test_diagnosis_loss += curr_batch_size * xEntropy_loss / N_test
                test_characteristic_loss += curr_batch_size * bce_loss / N_test
                
                predicted = y_hat[0].argmax(1)
                test_correct += (y[0]==predicted).sum().item()
            
        history['train_loss'].append(train_loss.item())
        history['train_diagnosis_loss'].append(train_diagnosis_loss.item())
        history['train_characteristic_loss'].append(train_characteristic_loss.item())
        history['train_accuracy'].append(train_correct/N_train)
        history['test_loss'].append(test_loss.item())
        history['test_diagnosis_loss'].append(test_diagnosis_loss.item())
        history['test_characteristic_loss'].append(test_characteristic_loss.item())
        history['test_accuracy'].append(test_correct/N_test)
        

        if save_path is not None and len(history['test_loss'])==1:
            torch.save(model,save_path)
        elif save_path is not None and history['test_loss'][-1]<np.min(history['test_loss'][:-1]):
            torch.save(model,save_path)
            print('Saved model with test loss: {}'.format(history['test_loss'][-1]))
        
    return history

#%%
data = Loader(img_size=[224,224],img_path=IMG_PATH)

RESULTS = {'test':[],'val':[]}

def evaluate(model,data):
    N = len(data)
    diagnosis_pred = np.zeros((N,6))
    char_pred = np.zeros((N,7))

    diagnosis_true = np.zeros(N)
    char_true = np.zeros_like(char_pred)

    for i in range(N):
        img,y,c=data[i]
        y_hat = model(img.view(1,3,224,224).to(device))
        y_hat_d = y_hat[0].cpu().detach().numpy()
        y = y.numpy()
        
        diagnosis_pred[i] = y_hat_d
        
        cp = torch.nn.Sigmoid()(y_hat[1]).cpu().detach().numpy()
        char_pred[i] =  cp
        
        diagnosis_true[i] = y
        char_true[i] = c.cpu().numpy()

    return diagnosis_true, char_true, diagnosis_pred, char_pred



# For each run, generate a different dataset. set random seed for data split to i
for i in range(N_RUNS):
    
    model = Model(6,7)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    
    train_set,test_set,val_set = splitData(data,0.6,0.2,0.2,seed=i)
    train_set.setTransform(train_transform)
    
    

    # Compute disease class weights
    diagnosis_train = list(np.unique(train_set.data['diagnosis']))
    diagnosis_count = torch.zeros(len(diagnosis_train))
    for i in range(len(diagnosis_train)):
        diagnosis_count[i] = (train_set.data['diagnosis']==diagnosis_train[i]).sum()

    diagnosis_frac = diagnosis_count/diagnosis_count.sum()
    diagnosis_weight = (1-diagnosis_frac)/(1-diagnosis_frac).sum()

    # Compute characteristic weights for unbalanced data
    idx=0
    characteristic_count = torch.zeros(7)
    for key in train_set.data.keys()[3:-1]:
        characteristic_count[idx] = (train_set.data[key]>0).sum()
        idx+=1
    characteristic_weight = (1-characteristic_count/len(train_set)).to(device)
    characteristic_weight = (torch.ones(7)*0.5).to(device)
    characteristic_names = train_set.data.keys()[3:-1].tolist()
    
    
    # Train the model
    history = train(model,train_set,test_set,lossFunction,optimizer,diagnosis_weight,characteristic_weight,
                        num_epochs=NUM_EPOCHS,batch_size=16,num_workers=8,save_path='temp_net')
    
    
    # Load the best model
    model = torch.load('temp_net',map_location='cpu')
    model.eval()
    model.to(device)
    
    # Evaluate the model
    test_diag_true,test_char_true,test_diag_pred,test_char_pred = evaluate(model,test_set)
    val_diag_true,val_char_true,val_diag_pred,val_char_pred = evaluate(model,val_set)
    RESULTS['test'].append([test_diag_true,test_char_true,test_diag_pred,test_char_pred ])
    RESULTS['val'].append([val_diag_true,val_char_true,val_diag_pred,val_char_pred ])
    
    
    os.remove('temp_net')
    
    
    
# Save the evaluation file
with open('results.pickle','wb') as handle:
    pickle.dump(RESULTS,handle)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    