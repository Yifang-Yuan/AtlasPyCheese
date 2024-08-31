#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 23:13:51 2024

@author: zhumingshuai
"""
import pandas as pd
import os
import re

class key_trail:
    def __init__(self,cold,sync,atlas):
        self.cold = cold
        self.sync = sync
        self.atlas = atlas

class cold_file:
    def __init__(self,path):
        self.df = pd.read_excel(path)
    
    def ObtainKeyTrails(self,folder):
        for filename in os.listdir(folder):
            #in this case I write down trails with an Atlas recording as the filename of an empty docx document
            if filename.endswith('.docx'):
                self.key_index = re.findall(r'\d+', filename)[0]
        self.keynum = []
        
        pre_index = -1
        current_index = 0
        
        #aiming to deal with more than 10 trails
        for i in range (len(self.key_index)):
            current_index += int(self.key_index[i])
            if current_index > pre_index:
                pre_index = current_index
                self.keynum.append(current_index)
                current_index = 0
            else:
                current_index *= 10
        print(self.df.iloc[5])
        self.keydf = pd.DataFrame()
        for i in  self.keynum:
            self.keydf = pd.concat([self.keydf, self.df.iloc[[i]]], ignore_index=True)
        print(self.keydf)
            
    def Link2Sync (self,folder):
        for filename in os.listdir(folder):
            self.keysync = []
            if 'sync' in filename:
                ID = self.key_index = re.findall(r'\d+', filename.split('sync')[1])[0]
                if ID in self.keynum:
                    keysync.append(os.path.join(folder, filename)
    
    def Link2Atlas (self,folder):
        for filename in os.listdir(folder):
            self.keysync = []
            if 'sync' in filename:
                ID = self.key_index = re.findall(r'\d+', filename.split('sync')[1])[0]
                if ID in self.keynum:
                    keysync.append(os.path.join(folder, filename)
                    

def MainFunction (input_format_df):
    return

a = cold_file('/Users/zhumingshuai/Desktop/Programming/Data/Atlas/Sample/Training_Data_Day1.xlsx')
a.ObtainKeyTrails('/Users/zhumingshuai/Desktop/Programming/Data/Atlas/Sample')
b = a.keydf