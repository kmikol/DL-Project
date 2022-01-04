# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:06:32 2021

@author: Kamil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

labels = pd.read_csv('dermx_labels.csv')
diseases_characteristics = pd.read_csv('diseases_characteristics.csv')

# Replace NaN with string "irrelevant"
labels['area'] = labels['area'].fillna('irrelevant')

diseases_characteristics = diseases_characteristics.rename(columns={'Unnamed: 0': 'disease'})

#%%

diagnosis = np.unique(labels['diagnosis'])
areas = np.unique(labels['area'])

data_stats = {}

for d in diagnosis:
    N = (labels['diagnosis']==d).sum()
    
    N_scale = ((labels['diagnosis']==d) & (labels['scale']==1)).sum()
    N_plaque = ((labels['diagnosis']==d) & (labels['plaque']==1)).sum()
    N_pustule = ((labels['diagnosis']==d) & (labels['pustule']==1)).sum()
    N_patch = ((labels['diagnosis']==d) & (labels['patch']==1)).sum()
    N_papule = ((labels['diagnosis']==d) & (labels['papule']==1)).sum()
    N_dermatoglyph_disruption = ((labels['diagnosis']==d) & (labels['dermatoglyph_disruption']==1)).sum()
    N_open_comedo = ((labels['diagnosis']==d) & (labels['open_comedo']==1)).sum()
    
    data_stats[d] = {'N':N, 'scale':N_scale/N, 'plaque':N_plaque/N, 'pustule':N_pustule/N,
                     'patch':N_patch/N,'papule':N_papule/N,'dermatoglyph_disruption':N_dermatoglyph_disruption/N,
                     'open_comedo':N_open_comedo/N}
    
    for a in areas:
        N_area = ((labels['diagnosis']==d) & (labels['area']==a)).sum()
        data_stats[d][a] = N_area/N

#%% Plot number of samples for each disease in the dataset
counts=[]
for i in range(len(diagnosis)):
    counts.append(data_stats[diagnosis[i]]['N'])

plt.bar(diagnosis,counts)
plt.ylabel('Number of images')
plt.title('')
plt.xticks(rotation=90)
plt.grid()
plt.show()

#%% Plot feature distribution for each disease

for i in range(len(diagnosis)):
    
    #plt.subplot(len(diagnosis),1,i+1)
    bars = [*data_stats[diagnosis[i]].values()][1:]
    plt.bar([*data_stats[diagnosis[0]].keys()][1:],bars)
    plt.xticks(rotation=90)
    plt.title(f"{diagnosis[i]}, N: {data_stats[diagnosis[i]]['N']}")
    plt.ylabel('Fraction of samples where the feature occurs')
    plt.grid()
    plt.show()
    
    
    

    
#%% Try to classify the data using some simple metrics
    
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score

w_area = 3 # relative importance of the area feature (seems to be more important than others)

dc_np = diseases_characteristics.to_numpy()
data_np = labels.to_numpy()

pred_disease = np.empty(data_np.shape[0],dtype='object')

for i in range(data_np.shape[0]):
    
    sample_feature = data_np[i,2:-1]
    best_score = 0
    
    for j in range(dc_np.shape[0]):
        
        disease = dc_np[j,0]
        
        disease_feature = dc_np[j,1:-1]
        
        area_match = int(data_np[i,-1] == dc_np[j,-1])
        if data_np[i,-1]=='irrelevant':
            area_match = 1
        
        v1 = np.append(disease_feature,[area_match]*w_area).reshape(1,-1)
        v2 = np.append(sample_feature, [1]*w_area).reshape(1,-1)
        
        cs = cosine_similarity(v1,v2)
        #cs = jaccard_score(v1.astype(bool).flatten(),v2.astype(bool).flatten(),average='binary')
        if best_score<cs:
            best_score=cs
            pred_disease[i] = disease
        
        
        
#%% Plot confusion matrix for the simple classification
from sklearn.metrics import confusion_matrix
import seaborn as sn


columns = [*np.unique(data_np[:,1])]

confm = confusion_matrix(data_np[:,1], pred_disease)
df_cm = pd.DataFrame(confm, index=columns, columns=columns)

ax = sn.heatmap(df_cm, cmap='Blues', annot=True,cbar=False)
plt.title(f'Accuracy: {round(np.diag(confm).sum()/confm.sum()*100,2)}%')
plt.ylabel('True')
plt.xlabel('Predicted')
