#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 14:26:16 2021

@author: s202818
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle
import torch
from sklearn.naive_bayes import BernoulliNB
from data_loader import Loader
import pandas as pd


with open('results_50ep_weighted.pickle','rb') as handle:
    results = pickle.load(handle)
    
test_results = results['test']
val_results = results['val']

def diagnosisCorrection(diagnosis_pred,characteristic_pred,correction_dict):
    correction_model, alpha = correction_dict['model'], correction_dict['alpha']
    softmax = torch.nn.Softmax(dim=1)
    diagnosis_pred_softmax = softmax(torch.tensor(diagnosis_pred))
    diagnosis_pred_NB = correction_model.predict_proba(characteristic_pred>0.5)
    diagnosis_corr = softmax( (1-alpha) * diagnosis_pred_softmax + alpha * diagnosis_pred_NB )
    
    return diagnosis_corr

def getMetrics(true,pred):
    conf_mat = confusion_matrix(true,pred)
    acc= np.diag(conf_mat).sum()/conf_mat.sum()
    pre= np.diag(conf_mat) / conf_mat.sum(0)
    rec= np.diag(conf_mat) / conf_mat.sum(1)
    f1 = 2* pre*rec / (pre+rec)
    
    return acc,np.nan_to_num(pre,nan=0.0),np.nan_to_num(rec,nan=0.0),np.nan_to_num(f1,nan=0.0)

def processResults(results, correction_dict = None):
    diagnosis_metrics = []
    diagnosis_acc= []
    characteristic_metrics = []
    
    for result  in results:
    
        dt,ct,dp,cp = result
        
        if correction_dict is not None:
            dp = diagnosisCorrection(dp,cp,correction_dict)
        
        acc,pre,rec,f1 = getMetrics(dt,dp.argmax(1))
        diagnosis_acc.append(acc)
        diagnosis_metrics.append([pre,rec,f1])
        
        char = []
        for i in range(7):
            
            acc,pre,rec,f1 = getMetrics(ct[:,i],(cp>0.5)[:,i])
            char.append([acc,pre[0],rec[0],f1[0]])
        characteristic_metrics.append(np.array(char))
            
            
    diagnosis_acc = np.array(diagnosis_acc)
    diagnosis_metrics = np.array(diagnosis_metrics)
    characteristic_metrics = np.swapaxes(np.array(characteristic_metrics),1,2)
    
    return diagnosis_acc, diagnosis_metrics, characteristic_metrics



#%% Fit a bernoulli naive bayes model to use for diagnosis correction

train_np = Loader(img_size=[224,224],img_path=None).data.to_numpy()[:,1:-1]
train_np[:,-1] = np.clip(train_np[:,-1], 0, 1) # The last value should be between 0 and 1, not 0 and 8

disease_characteristics_np = pd.read_csv('diseases_characteristics_sorted.csv',delimiter=',').to_numpy()


modelNB_dc = BernoulliNB()
modelNB_dc.fit(disease_characteristics_np[:,1:-1],disease_characteristics_np[:,0])
print(f'Bayes Classifier (disease char. based) val score: {modelNB_dc.score(train_np[:,1:],train_np[:,0])}')


modelNB = BernoulliNB()
modelNB.fit(train_np[:,1:],train_np[:,0])
print(f'Bayes Classifier (data based) val score: {modelNB.score(train_np[:,1:],train_np[:,0])}')

correction_dict = {'model':modelNB,'alpha':0}





#%% Choose the best alpha parameter to combine predicted diagnosis and correction
As = np.arange(0,1,0.01)
accuracies_test = np.zeros((len(As),10))
accuracies_val = np.zeros((len(As),10))
i=0

for a in As:
    correction_dict['alpha']=a

    diagnosis_acc_test, _,_= processResults(test_results, correction_dict=correction_dict)
    diagnosis_acc_val, _,_= processResults(val_results, correction_dict=correction_dict)
    accuracies_test[i]=diagnosis_acc_test
    accuracies_val[i]=diagnosis_acc_val
    i+=1
    

plt.figure(figsize=(4,4))
plt.plot(As,accuracies_test.mean(1),label='Validation') 
plt.plot(As,accuracies_val.mean(1),label='Test')
plt.grid()
plt.xlabel(r'$\alpha$')
plt.ylabel('Accuracy')
plt.title('Diagnosis Correction')
plt.legend()
plt.tight_layout()
#plt.ylim([0.5,1])
plt.savefig('figures/diagnosis_correction.eps')



#%%
correction_dict['alpha'] = As[np.argmax(accuracies_test.mean(0))]
diagnosis_acc, diagnosis_metrics, characteristic_metrics = processResults(val_results, correction_dict=correction_dict)


#%% Plot
diagnosis_str = np.array(['Acne','Actinic Keratosis','Psoriasis','Seborrheic Dermatitis','Viral Warts','Vitiligo'])
characteristic_str = ['Scale','Plaque','Pustule','Patch','Papule','Derm. Disruption','Open Comedo']
metric_str = ['Accuracy','Precision','Recall','F1 Score']

print('Diagnosis Metric \n')
print(f'Diagnosis accuracy: {diagnosis_acc.mean()}, std: {diagnosis_acc.std()} \n')

for i in range(1,4,1):
    print(f'{metric_str[i]}')
    for j in range(6):
        print(f'{diagnosis_str[j]}: \t {np.round(diagnosis_metrics[:,i-1,j].mean(0),4)}, \
              std: {np.round(diagnosis_metrics[:,i-1,j].std(0),4)}')
    print('\n')
    

print('Characteristic metrics \n')
for i in range(0,4,1):
    print(f'{metric_str[i]}')
    for j in range(7):
        print(f'{characteristic_str[j]}: \t {np.round(characteristic_metrics[:,i,j].mean(0),4)},  \
              std: {np.round(characteristic_metrics[:,i,j].std(0),4)}')
    print('\n')

#%%

errbar_dict = {'marker':'^','fmt':' ','mec':'k','ms':10,'capsize':10,
               'capthick':3,'ecolor':'tab:Red','elinewidth':1}

for i in range(1,4,1):

    plt.figure(figsize=(5,5))
    plt.errorbar(diagnosis_str, diagnosis_metrics[:,i-1,:].mean(0),diagnosis_metrics[:,i-1,:].std(0),**errbar_dict)
    for j in range(len(diagnosis_str)):
        plt.scatter([diagnosis_str[j]]*diagnosis_metrics.shape[0],diagnosis_metrics[:,i-1,j],
                    marker='x',color='tab:Blue')        
    plt.xticks(rotation=60)
    plt.title(metric_str[i])
    plt.grid()
    plt.ylim([0.5,1])
    plt.tight_layout()
    plt.savefig(f'figures/diagnosis_{metric_str[i]}.eps')

#%%
for i in range(0,4,1):

    plt.figure(figsize=(5,5))
    plt.errorbar(characteristic_str, characteristic_metrics[:,i,:].mean(0),characteristic_metrics[:,i,:].std(0),**errbar_dict)
    for j in range(len(characteristic_str)):
        plt.scatter([characteristic_str[j]]*characteristic_metrics.shape[0],characteristic_metrics[:,i,j],
                    marker='x',color='tab:Blue')
    plt.xticks(rotation=60)
    plt.title(metric_str[i])
    plt.grid()
    plt.ylim([0.5,1])
    plt.tight_layout()
    plt.savefig(f'figures/characteristics_{metric_str[i]}.eps')
    
