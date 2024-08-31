#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 23:13:51 2024

@author: zhumingshuai
"""
import pandas as pd
import os
import re

input_df = {
    'sync_tag':'sync',
    'sync_parent_folder_tag':'Bonsai',
    'atlas_parent_folder_tag':'Atlas_result',
    'key_suffix':'.docx',
    'atlas_z_filename':'Zscore_traceAll.csv'
    }

class key_trail:
    def __init__(self,cold,sync,atlas):
        self.cold = cold
        self.sync = sync
        self.atlas = atlas

class cold_file:
    def __init__ (self,cold_file_path,sync_folder,atlas_folder,input_df):
        self.df = pd.read_excel(cold_file_path)
        for filename in os.listdir(sync_folder):
            #in this case I write down trails with an Atlas recording as the filename of an empty docx document
            if filename.endswith(input_df['key_suffix']):
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
        
        # self.keydf = pd.DataFrame()
        # for i in  self.keynum:
        #     self.keydf = pd.concat([self.keydf, self.df.iloc[[i]]], ignore_index=True)
        # print(self.keydf)
        
        self.key_trails = []
        for index, i in enumerate(self.keynum):
            for filename in os.listdir(sync_folder):
                if input_df['sync_tag'] in filename:
                    ID = re.findall(r'\d+', filename.split(input_df['sync_tag'])[1])[0]
                    if int(ID) == i:
                        sync_file = pd.read_csv(os.path.join(sync_folder, filename))
                        
            for foldername in os.listdir(atlas_folder):
                if input_df['atlas_parent_folder_tag'] in foldername:
                    ID = re.findall(r'\d+', foldername.split(input_df['atlas_parent_folder_tag'])[1])[0]
                    if int(ID) == index+1:
                        folder_path = os.path.join(atlas_folder, foldername)
                        atlas_file = pd.read_csv(os.path.join(folder_path, input_df['atlas_z_filename']))
            
            self.key_trails.append(key_trail(self.df.iloc[[i]], sync_file, atlas_file))
            


def MainFunction (input_format_df):
    return

atlas_folder = '/Users/zhumingshuai/Desktop/Programming/Data/Atlas/Sample/Atlas/'
sync_folder = '/Users/zhumingshuai/Desktop/Programming/Data/Atlas/Sample/'
cold_folder = '/Users/zhumingshuai/Desktop/Programming/Data/Atlas/Sample/Training_Data_Day1.xlsx'
a = cold_file(cold_folder,sync_folder,atlas_folder,input_df)