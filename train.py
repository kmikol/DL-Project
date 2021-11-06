# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 10:33:12 2021

@author: Kamil
"""

import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import copy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'The model will run on: {device}')



#%% Load data and split it to tran,test and val sets
from data_loader import Loader, splitData

img_path = "C:\\Users\\Kamil\\OneDrive - Danmarks Tekniske Universitet\\DTU_courses\\Deep_Learning\\project\\images\\"

data = Loader(img_size=[224,224],img_path=img_path)
train_set,test_set,val_set = splitData(data,0.6,0.2,0.2)
print(f'Number of images:\nTrain: {len(train_set)}\nTest: {len(test_set)}\nVal: {len(val_set)}')

#%% Check if all classes are represented in train, test and val set
diagnosis_train = list(np.unique(train_set.data['diagnosis']))
diagnosis_test = list(np.unique(test_set.data['diagnosis']))
diagnosis_val = list(np.unique(val_set.data['diagnosis']))

diagnosis_train.sort()
diagnosis_test.sort()
diagnosis_val.sort()

if len(diagnosis_train) == len(diagnosis_test) == len(diagnosis_val):
    for i in range(len(diagnosis_train)):
        if diagnosis_train[i] == diagnosis_test[i] == diagnosis_val[i]:
            pass
        else:
            raise Exception('Not all classes are present in train, test or val set')
else:
    raise Exception('Not all classes are present in train, test or val set')

#%% Compute disease weights for unbalanced data

# Compute disease class weights
diagnosis_count = torch.zeros(len(diagnosis_train))
for i in range(len(diagnosis_train)):
    diagnosis_count[i] = (train_set.data['diagnosis']==diagnosis_train[i]).sum()

diagnosis_frac = diagnosis_count/diagnosis_count.sum()
diagnosis_weight = (1-diagnosis_frac)/(1-diagnosis_frac).sum()

#%% Compute feature weights for unbalanced data
i=0
feature_count = torch.zeros(7)
for key in train_set.data.keys()[3:-1]:
    feature_count[i] = (train_set.data[key]>0).sum()
    i+=1

feature_weight = (1-feature_count/len(train_set)).to(device)

feature_names = train_set.data.keys()[3:-1].tolist()

#%% Set train set augmentation

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
train_set.setTransform(train_transform)


#%% Show a few training pictures

def t2i(img):
    return torch.swapaxes(torch.swapaxes(img,1,2),0,2)

img,diagnosis,feature = val_set[0]
plt.imshow(t2i(img))
plt.show()

#%% Model setup

class Model(torch.nn.Module):
    def __init__(self,num_diagnosis,num_features):
        super(Model,self).__init__()
        
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_diagnosis+num_features)
        
        for param in model.parameters():
            param.requires_grad=True
        
        self.model = model
        self.num_diagnosis = num_diagnosis
        self.num_features = num_features
        
    def forward(self,x):
        
        y = self.model(x)
        diagnosis_enc = y[:,:self.num_diagnosis]
        feature_enc = y[:,self.num_diagnosis:]
        
        return diagnosis_enc, feature_enc
  
def bce(y_hat,y,weights = feature_weight ):
    p = torch.nn.Sigmoid()(torch.clip(y_hat,-10,10))
    loss = -torch.mean(feature_weight*y*torch.log(p) + (1-feature_weight)*(1-y)*torch.log(1-p))
    return loss
      
def lossFunction(y_hat,y):
    
    y_hat_diagnosis = y_hat[0]
    y_hat_feature = y_hat[1]
    y_diagnosis = y[0]
    y_feature = y[1]
        
    
    xEntropy = torch.nn.CrossEntropyLoss(weight=diagnosis_weight.to(device),reduce='mean')
    #bce = torch.nn.BCELoss(reduce='mean')
    sigmoid = torch.nn.Sigmoid()
    xEntropy_loss = xEntropy(y_hat_diagnosis,y_diagnosis)
    bce_loss = bce(y_hat_feature,y_feature)
    
    if bce_loss<0:
        print(sigmoid(y_hat_feature))
        print(y_feature)
        print(bce_loss)
    
    return xEntropy_loss, bce_loss

model = Model(6,7)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)




#%% Train

def train(model,train_set,test_set,lossFunction,optimizer,num_epochs=10,batch_size=64,num_workers=16, save_path=None):
    
    history = {'train_loss':[],
               'train_diagnosis_loss':[],
               'train_feature_loss':[],
               'train_accuracy':[],
               'test_loss':[],
               'test_diagnosis_loss':[],
               'test_feature_loss':[],
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
        train_feature_loss = 0
        train_correct = 0
        
        test_loss = 0
        test_diagnosis_loss = 0
        test_feature_loss = 0
        test_correct = 0
        
        model.train()
        for img,disease,feature in tqdm(train_iterator, desc='train: '):
            X = img.to(device).float()
            y = [disease.to(device), feature.to(device)]
            
            curr_batch_size = disease.shape[0]
            
            optimizer.zero_grad()
            y_hat = model(X)
            xEntropy_loss, bce_loss = lossFunction(y_hat,y)
            loss = xEntropy_loss + bce_loss
            loss.backward()
            optimizer.step()
            
            train_loss += curr_batch_size * loss / N_train
            train_diagnosis_loss += curr_batch_size * xEntropy_loss / N_train
            train_feature_loss += curr_batch_size * bce_loss / N_train
            
            predicted = y_hat[0].argmax(1)
            train_correct += (y[0]==predicted).sum().item()
        
        model.eval()
        for img,disease,feature in tqdm(test_iterator, desc='test: '):
            
            with torch.no_grad():
                X = img.to(device).float()
                y = [disease.to(device), feature.to(device)]
                
                curr_batch_size = disease.shape[0]
            
                y_hat = model(X)
                xEntropy_loss, bce_loss = lossFunction(y_hat,y)
                loss = xEntropy_loss + bce_loss
            
                test_loss += curr_batch_size * loss / N_test
                test_diagnosis_loss += curr_batch_size * xEntropy_loss / N_test
                test_feature_loss += curr_batch_size * bce_loss / N_test
                
                predicted = y_hat[0].argmax(1)
                test_correct += (y[0]==predicted).sum().item()
            
        history['train_loss'].append(train_loss.item())
        history['train_diagnosis_loss'].append(train_diagnosis_loss.item())
        history['train_feature_loss'].append(train_feature_loss.item())
        history['train_accuracy'].append(train_correct/N_train)
        history['test_loss'].append(test_loss.item())
        history['test_diagnosis_loss'].append(test_diagnosis_loss.item())
        history['test_feature_loss'].append(test_feature_loss.item())
        history['test_accuracy'].append(test_correct/N_test)
        

        if save_path is not None and len(history['test_loss'])==1:
            torch.save(model,save_path)
        elif save_path is not None and history['test_loss'][-1]<np.min(history['test_loss'][:-1]):
            torch.save(model,save_path)
            print('Saved model with test loss: {}'.format(history['test_loss'][-1]))
        
    return history



history = train(model,train_set,test_set,lossFunction,optimizer,
                num_epochs=100,batch_size=8,num_workers=0,save_path='resnet18_1')


#%% Plot history

e = np.arange(1,len(history['train_loss'])+1)
plt.plot(e,history['train_loss'],label='train')
plt.plot(e,history['test_loss'],label='test')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend(); plt.grid(); plt.show()


plt.plot(e,history['train_accuracy'],label='train')
plt.plot(e,history['test_accuracy'],label='test')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend(); plt.grid(); plt.show()

#%%
e = np.arange(1,len(history['train_loss'])+1)
plt.plot(e,history['train_diagnosis_loss'],label='train diagnosis')
plt.plot(e,history['train_feature_loss'],label='train feature')
plt.plot(e,history['test_diagnosis_loss'],label='test diagnosis')
plt.plot(e,history['test_feature_loss'],label='test feature')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend(); plt.grid(); plt.show()




#%%
def t2i(I):
    return torch.swapaxes(torch.swapaxes(I,1,2),0,2)

for i in range(5):
    i,d,f=val_set[i]
    
    y_hat = model(i.view(1,3,224,224).to(device))
    y_hat_feat = np.round(torch.nn.Sigmoid()(y_hat[1]).cpu().detach().numpy(),2)[0]
    f = f.cpu().numpy()
    diagnosis_pred = diagnosis_train[y_hat[0].argmax().cpu().numpy()]
    diagnosis = diagnosis_train[d]
    
    plt.title(f'Dermatologist: {diagnosis}, \n Model: {diagnosis_pred} \n {np.round(f,2)} \n {y_hat_feat}')
    plt.imshow(t2i(i))
    plt.show()
    

#%%
correct=0
N = 0
for i in range(len(val_set)):
    i,y,f=val_set[i]
    y_hat = model(i.view(1,3,224,224).to(device))
    y_hat = y_hat[0].argmax().cpu().numpy()
    y = y.numpy()
    
    N+=1
    if y==y_hat:
        correct+=1
    

print(correct/N)

#%%

def saliencyMap(model,optimizer,img,label,num_imgs,std=0.05,mean=0.0):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    imgs = torch.zeros(num_imgs,img.shape[0],img.shape[1],img.shape[2])
    y = (torch.ones(num_imgs,dtype=torch.long) * label).to(device)
    
    for i in range(num_imgs):
        imgs[i]=img
    
    # add noise
    imgs += torch.rand(imgs.size()) * std + mean
    
    model.to(device)
    imgs = imgs.to(device)
    imgs = imgs.requires_grad_()
    y_hat = model(imgs.float())[0]
    
    optimizer.zero_grad()
    lossFunction =  torch.nn.CrossEntropyLoss()
    loss = lossFunction(y_hat,y)
    loss.backward()
    
    saliency_map,_ = torch.max(imgs.grad.data.abs(),dim=1)
    saliency_map = torch.mean(saliency_map,dim=0)
    
    quant = torch.quantile(saliency_map,torch.tensor([0.02,0.98],device=device))
    saliency_map = torch.clip(saliency_map,min=quant[0],max=quant[1])
    
    
    plt.subplot(1,2,2)
    plt.imshow(saliency_map.detach().cpu().numpy(),cmap='jet')
    plt.grid();
    
    plt.subplot(1,2,1)
    plt.imshow(t2i(img))
    plt.grid(); plt.tight_layout(); plt.show()
    
    
# This doesnt work properly
def saliencyMapAll(model,optimizer,img,label,features,num_imgs,std=0.05,mean=0.0):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    imgs = torch.zeros(num_imgs,img.shape[0],img.shape[1],img.shape[2])
    y = (torch.ones(num_imgs,dtype=torch.long) * label).to(device)
    
    for i in range(num_imgs):
        imgs[i]=img
    
    # add noise
    imgs += torch.rand(imgs.size()) * std + mean
    
    model.to(device)
    imgs = imgs.to(device)
    imgs = imgs.requires_grad_()
    y_hat =  model(imgs.float())
    y_hat_d = y_hat[0]
    y_hat_f = y_hat[1]
    
    optimizer.zero_grad()
    lossFunction =  torch.nn.CrossEntropyLoss()
    loss = lossFunction(y_hat_d,y)
    loss.backward()
    
    saliency_map,_ = torch.max(imgs.grad.data.abs(),dim=1)
    saliency_map = torch.mean(saliency_map,dim=0)
    
    quant = torch.quantile(saliency_map,torch.tensor([0.02,0.98],device=device))
    saliency_map = torch.clip(saliency_map,min=quant[0],max=quant[1])
    
    plt.figure(figsize=(20,20))
    
    plt.subplot(3,3,2)
    plt.imshow(saliency_map.detach().cpu().numpy(),cmap='jet')
    plt.grid();
    
    plt.subplot(3,3,1)
    plt.imshow(t2i(img))
    plt.grid(); plt.tight_layout();

    
    for i in range(0,7):
        
        optimizer.zero_grad()
        
        imgs = imgs.to(device)
        imgs = imgs.requires_grad_()
        y_hat =  model(imgs.float())
        y_hat_d = y_hat[0]
        y_hat_f = y_hat[1]
        
        features_true = torch.ones_like(y_hat_f)*features.to(device).detach()
        features_predicted = torch.ones_like(y_hat_f)*10*(2*features-1).to(device).detach()
        features_predicted[:,i]=y_hat_f[:,i]
        
        
        loss = bce(features_predicted,features_true)
        loss.backward()
        
        saliency_map,_ = torch.max(imgs.grad.data.abs(),dim=1)
        saliency_map = torch.mean(saliency_map,dim=0)
        
        quant = torch.quantile(saliency_map,torch.tensor([0.02,0.98],device=device))
        saliency_map = torch.clip(saliency_map,min=quant[0],max=quant[1])
        
        
        plt.subplot(3,3,3+i)
        plt.imshow(saliency_map.detach().cpu().numpy(),cmap='jet')
        plt.title(feature_names[i])
        plt.grid();
    
    plt.show()

for i in range(10):
    img,d,f = val_set[i]
    saliencyMap(model,optimizer,img,d,50,std=0.1)
    
    #saliencyMapAll(model,optimizer,img,d,f,50,std=0.1)
    
    
#%%

for i in range(8):
    print(bce(a_[i],b_[i]))