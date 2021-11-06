# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:06:55 2021

@author: Kamil
"""

import csv
from PIL import Image


relative_path = "C:\\Users\\Kamil\\OneDrive - Danmarks Tekniske Universitet\\DTU_courses\\Deep_Learning\\project\\images\\"

keepers=[]

with open("dermx_labels.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
        
        if i==0: # This is the header
            keepers.append(line)
        
        # Try to load the image, keep the entry if it works
        try:
            img= Image.open(relative_path+line[0].split(',')[0]+'.jpeg')
            keepers.append(line)
        except FileNotFoundError:
            pass
        
with open('dermx_labels_filtered.csv', 'w', newline='\n') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    for keeper in keepers:
        csvwriter.writerow(keeper)