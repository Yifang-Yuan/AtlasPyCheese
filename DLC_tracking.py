# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:20:45 2024

@author: s2764793
"""
import os
import pandas as pd
import re

parameter = {
    'split_tag':'cam',
    'tracking_file_sufix': '.csv',
    'tracking_file_tag': 'DLC'
    }

class frame:
    def __init__(self,head,shoulder,bottom):
        self.head = head
        self.shoulder = shoulder
        self.bottom = bottom

class trace:
    def __init__(self,file,ID,day):
        self.file = file
        self.ID = ID
        self.day = day
        for i in range (file.shape[0]):
            if file.iloc[i,0] == 'bodyparts':
                self.bdp_index = i
            if file.iloc[i,0] == 'coords':
                self.xy_index = i
        for j in range (file.shape[1]):
            if 
        
        
class dayx:
    def __init__(self,parameter,folder,day):
        self.day = day
        for filename in os.listdir(folder):
            if filename.endswith(parameter['tracking_file_sufix']) and parameter['tracking_file_tag'] in filename:
                file_path = os.path.join(folder,filename)
                ID = int(re.findall(r'\d+', filename.split(parameter['split_tag'])[1])[0])
                file = pd.read_csv(file_path)
                t = trace(file,ID,self.day)
                
                
folder = '//cmvm.datastore.ed.ac.uk/cmvm/sbms/users/s2764793/Win7/Desktop/DLC/'
a = dayx(parameter,folder,1)

                
                