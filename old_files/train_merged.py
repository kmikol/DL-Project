# -*- coding: utf-8 -*-


import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import copy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'The model will run on: {device}')

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#%% Load data and split it to tran,test and val sets
from data_loader import Loader, splitData, get_diagnosis_with_feature

#img_path = "C:\\Users\\Kamil\\OneDrive - Danmarks Tekniske Universitet\\DTU_courses\\Deep_Learning\\project\\images\\"
img_path = "C:\\Users\\Micha\\OneDrive\\DTU -Kandidat\\3_Semester\\Deep\\images\\"

data = Loader(img_size=[224,224],img_path=img_path)
train_set,test_set,val_set = splitData(data,0.6,0.2,0.2,seed=0)
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

#%% Compute characteristic weights for unbalanced data
i=0
characteristic_count = torch.zeros(7)
for key in train_set.data.keys()[3:-1]:
    characteristic_count[i] = (train_set.data[key]>0).sum()
    i+=1

characteristic_weight = (1-characteristic_count/len(train_set)).to(device)

characteristic_names = train_set.data.keys()[3:-1].tolist()

#%%
feature_order = ['scale','plaque','pustule','patch','papule','dermatoglyph_disruption','open_comedo']
diagnosis_order =  ['acne', 'actinic_keratosis', 'psoriasis', 'seborrheic_dermatitis', 'viral_warts', 'vitiligo']
  
diagnosis_feature = get_diagnosis_with_feature(img_path,feature_order,diagnosis_order)


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

img,diagnosis,characteristic = train_set[0]
plt.imshow(t2i(img))
plt.show()

#%% Model setup

class Model(torch.nn.Module):
    def __init__(self,num_diagnosis,num_characteristics,base_model = 'resnet'):
        super(Model,self).__init__()
        
        if base_model == 'densenet':
            model = models.densenet121(pretrained=True)
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_diagnosis+num_characteristics)
        
        elif base_model == 'resnet':
            model = models.resnet18(pretrained=True)
            model.fc = torch.nn.Linear(model.fc.in_features, num_diagnosis+num_characteristics)
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
  
def svm(y_hat,y,y_diagnosis,diagnosis_feature=diagnosis_feature):
    p = torch.nn.Sigmoid()(torch.clip(y_hat,-10,10))
    xBCE = torch.nn.BCELoss(reduce="mean")
    loss = xBCE(p,diagnosis_feature[y_diagnosis])
    #print(loss)

    # Also considered HingeEmbeddingLoss or similar, not that good
    #xHinge = HingeEmbeddingLoss(margin=1.0, size_average=None, reduce=None, reduction='mean')

    loss = bce(y_hat,diagnosis_feature[y_diagnosis])
    #print(loss)
    return loss  
  
def bce(y_hat,y,weights = characteristic_weight ):
    p = torch.nn.Sigmoid()(torch.clip(y_hat,-10,10))
    loss = -torch.mean(characteristic_weight*y*torch.log(p) + (1-characteristic_weight)*(1-y)*torch.log(1-p))
    return loss
      
def lossFunction(y_hat,y):
    
    y_hat_diagnosis = y_hat[0]
    y_hat_characteristic = y_hat[1]
    y_diagnosis = y[0]
    y_characteristic = y[1]
        
    
    xEntropy = torch.nn.CrossEntropyLoss(weight=diagnosis_weight.to(device),reduce='mean')
    #bce = torch.nn.BCELoss(reduce='mean')
    sigmoid = torch.nn.Sigmoid()
    xEntropy_loss = xEntropy(y_hat_diagnosis,y_diagnosis)
    bce_loss = bce(y_hat_characteristic,y_characteristic)
    
    svm_loss = svm(y_hat_characteristic,y_characteristic,y_diagnosis)
    
    if bce_loss<0:
        print(sigmoid(y_hat_characteristic))
        print(y_characteristic)
        print(bce_loss)
    
    return xEntropy_loss, bce_loss, svm_loss

model = Model(6,7)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)




#%% Train

def train(model,train_set,test_set,lossFunction,optimizer,num_epochs=10,batch_size=64,num_workers=16, save_path=None):
    
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
            xEntropy_loss, bce_loss, svm_loss = lossFunction(y_hat,y)
            loss = xEntropy_loss + bce_loss + svm_loss
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
                xEntropy_loss, bce_loss, svm_loss = lossFunction(y_hat,y)
                loss = xEntropy_loss + bce_loss + svm_loss
            
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

#%% Train the model
history = train(model,train_set,test_set,lossFunction,optimizer,
                    num_epochs=5,batch_size=16,num_workers=0,save_path='resnet18_3')

#%% Load pretrained model for evaluation
model = torch.load('testher_1')
model.eval()
model.to(device)


#%% Plot history
# Works only if the model was trained during the same session (history is not saved)
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

#%% Show the contribution of diagnosis and characteristics loss
e = np.arange(1,len(history['train_loss'])+1)
plt.plot(e,history['train_diagnosis_loss'],label='train diagnosis')
plt.plot(e,history['train_characteristic_loss'],label='train characteristic')
plt.plot(e,history['test_diagnosis_loss'],label='test diagnosis')
plt.plot(e,history['test_characteristic_loss'],label='test characteristic')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend(); plt.grid(); plt.show()




#%% Plot pictures with predictions
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
    

#%% Evaluate the network
N = len(val_set)
diagnosis_pred = np.zeros(N,dtype=int)
char_pred = np.zeros((N,7))

diagnosis_true = np.zeros_like(diagnosis_pred)
char_true = np.zeros_like(char_pred)

for i in range(len(val_set)):
    img,y,c=val_set[i]
    y_hat = model(img.view(1,3,224,224).to(device))
    y_hat_d = y_hat[0].argmax().cpu().numpy().item()
    y = y.numpy()
    
    diagnosis_pred[i] = y_hat_d
    
    cp = torch.nn.Sigmoid()(y_hat[1]).cpu().detach().numpy()
    char_pred[i] =  cp
    
    diagnosis_true[i] = y
    char_true[i] = c.cpu().numpy()

char_pred_b = char_pred>0.5
    



#%%

# Plot confusion matrix for the diagnosis
import seaborn as sb
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

conf_mat = confusion_matrix(diagnosis_true,diagnosis_pred)

df_cm = pd.DataFrame(conf_mat,index=[i for i in diagnosis_val],
                     columns=[i for i in diagnosis_val])

plt.figure(figsize=(5,5),dpi=300)
sb.heatmap(df_cm,annot=True,cmap='Blues',cbar=False)
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()

# Plot accuracy for the characteristics
char_acc = (char_true==char_pred_b).sum(0)/N
plt.bar(characteristic_names,char_acc)
plt.xticks(rotation=90)
plt.show()

# Get some more stats

FP = (conf_mat.sum(axis=0) - np.diag(conf_mat)).astype(float)  
FN = (conf_mat.sum(axis=1) - np.diag(conf_mat)).astype(float)
TP = (np.diag(conf_mat)).astype(float)
TN = (conf_mat.sum() - (FP + FN + TP)).astype(float)



Sensitivity = TP/(TP+FN)
Specificity = TN/(TN+FP) 

Recall = (TP/(TP+FN)).astype(float)  
Precision = (TP/(TP+FP)).astype(float)  

F1score = 2* ((Precision*Recall)/(Precision+Recall))
print(F1score)

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
def saliencyMapAll(model,optimizer,img,label,characteristics,num_imgs,std=0.05,mean=0.0):
    
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
        
        characteristics_true = torch.ones_like(y_hat_f)*characteristics.to(device).detach()
        characteristics_predicted = torch.ones_like(y_hat_f)*10*(2*characteristics-1).to(device).detach()
        characteristics_predicted[:,i]=y_hat_f[:,i]
        
        
        loss = bce(characteristics_predicted,characteristics_true)
        loss.backward()
        
        saliency_map,_ = torch.max(imgs.grad.data.abs(),dim=1)
        saliency_map = torch.mean(saliency_map,dim=0)
        
        quant = torch.quantile(saliency_map,torch.tensor([0.05,0.95],device=device))
        saliency_map = torch.clip(saliency_map,min=quant[0],max=quant[1])
        
        
        plt.subplot(3,3,3+i)
        plt.imshow(saliency_map.detach().cpu().numpy(),cmap='jet')
        plt.title(characteristic_names[i])
        plt.grid();
    
    plt.show()

for i in range(10):
    img,d,f = val_set[i]
    saliencyMap(model,optimizer,img,d,50,std=0.05)
    
    #saliencyMapAll(model,optimizer,img,d,f,50,std=0.1)
    
    
