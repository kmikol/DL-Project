# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:23:28 2021

@author: Kamil
"""





import torch
import pandas as pd
from PIL import Image
from copy import deepcopy
import numpy as np
from torchvision import transforms



class Loader(torch.utils.data.Dataset):
    
    
    def __init__(self, img_size, img_path ,transform = None):
        
        
        self.data = pd.read_csv('dermx_labels_filtered.csv',delimiter=',')
        self.transform = transform
        self.img_size = img_size
        self.img_path = img_path
        self.diagnosis_str = list(np.unique(self.data['diagnosis']))
        
        
    def __len__(self):
        
        return self.data.shape[0]
    
    
    def __getitem__(self,idx):
        
        img = Image.open(self.img_path+self.data['image_id'][idx]+".jpeg")
        diagnosis =  self.data['diagnosis'][idx]
        scale                   = self.data['scale'][idx]
        plaque                  = self.data['plaque'][idx]
        pustule                 = self.data['pustule'][idx]
        patch                   = self.data['patch'][idx]
        papule                  = self.data['papule'][idx]
        dermatoglyph_disruption = self.data['dermatoglyph_disruption'][idx]
        open_comedo             = self.data['open_comedo'][idx]
        
        area = self.data['area'][idx]
        
        
        features = torch.tensor([scale,plaque,pustule,patch,papule,dermatoglyph_disruption,open_comedo],dtype=torch.float)
        features=torch.clip(features,0,1) # quick fix
        
        img = img.resize(self.img_size)       
        img = transforms.ToTensor()(img)

        
        if self.transform is not None:
            img = self.transform(img)
        
        diagnosis_int = torch.tensor(self.diagnosis_str.index(diagnosis),dtype=torch.long)
        
        return img, diagnosis_int, features
    
    def setTransform(self,transform):
        self.transform = transform
    

def splitData(data,train,test,val,seed=0):
    
    
    if train+test+val != 1.0:
        raise Exception('train, test and val must sum to 1!')
        
    train_set = deepcopy(data)
    test_set = deepcopy(data)
    val_set = deepcopy(data)
    
   if seed is not None:
        np.random.seed(seed)
        rand_idx = np.random.choice(3,len(data),p=[train,test,val])
    else:
        rand_idx = np.random.choice(3,len(data),p=[train,test,val])
    
    
    if train!=0.0:
        train_set.data = train_set.data.loc[rand_idx==0]
        train_set.data = train_set.data.reset_index()
    else:
        train_set = None
    
    if test!=0.0:
        test_set.data = test_set.data.loc[rand_idx==1]
        test_set.data = test_set.data.reset_index()
    else:
        test_set = None
        
    if val!=0.0:
        val_set.data = val_set.data.loc[rand_idx==2]
        val_set.data = val_set.data.reset_index()
    else:
        val_set = None
    
    return train_set, test_set, val_set
    


