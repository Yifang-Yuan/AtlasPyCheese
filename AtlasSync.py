#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 23:13:51 2024

@author: zhumingshuai
"""
import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import AtlasFunction as af

input_format_df = {
    'sync_tag':'sync',
    'bonsai_folder_tag':'Bonsai',
    'atlas_folder_tag':'Atlas_Trial',
    'atlas_parent_folder_tag':'Atlas',
    'key_suffix':'.docx',
    'atlas_z_filename':'Zscore_trace.csv',
    'cold_folder_tag':'Cold_folder',
    'day_tag': 'Day',
    'key_ignore_time':120
    }

pfw = None

class key_trail:
    def __init__(self,cold,sync,atlas):
        self.cold = cold
        self.sync = sync
        self.atlas = atlas
        self.Synchronisation()
        self.smoothed_atlas = pd.DataFrame(af.smooth_signal(self.atlas[0],window_len=10),columns=[0])
        self.PlotSingleTrail()
        
    def Synchronisation (self):
        for i in range (len(self.sync)):
            if np.isnan(self.sync['Value.X'].iloc[i]) and np.isnan(self.sync['Value.Y'].iloc[i]):
                self.startframe_sync = i
                self.startframe_atlas = i*35
                self.starttime_sync = i/24
                print(self.startframe_sync)
                break
        global pfw
        w1t = self.cold.loc[0,'well1time_s']
        w2t = self.cold.loc[0,'well2time_s']
        self.starttime_cold = self.cold.loc[0,'startingtime_s']
        if w1t >= input_format_df['key_ignore_time']:
            w1t = np.nan
        if w2t >= input_format_df['key_ignore_time']:
            w2t = np.nan
        if pfw == 1:
            self.pfw_enter = w1t
            self.lpfw_enter = w2t
        elif pfw == 2:
            self.pfw_enter = w2t
            self.lpfw_enter = w1t
        if (not np.isnan(self.cold.loc[0,'firstwellreached'])) and int(self.cold.loc[0,'firstwellreached'])==pfw:
            self.pfw_leave = self.cold.loc[0,'leftfirstwell_s']
        else:
            self.pfw_leave = np.nan
        if (not np.isnan(self.cold.loc[0,'firstwellreached'])) and int(self.cold.loc[0,'firstwellreached'])!=pfw:
            self.lpfw_leave = self.cold.loc[0,'leftfirstwell_s']
        else:
            self.lpfw_leave = np.nan
        return
            
    def PlotSingleTrail(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.smoothed_atlas.index/840,self.smoothed_atlas[0])
        ax.set_title(self.cold.loc[0,'name'])
        if not (np.isnan(self.pfw_enter)):
            if (self.pfw_enter+self.starttime_cold-self.starttime_sync < 30):
                ax.axvline(x=self.pfw_enter+self.starttime_cold-self.starttime_sync, color='r', linestyle='--', label='preferred_well_enter_time')
            if (self.pfw_leave+self.starttime_cold-self.starttime_sync < 30):
                ax.axvline(x=self.pfw_leave+self.starttime_cold-self.starttime_sync, color='g', linestyle='--', label='preferred_well_leave_time')
        if not (np.isnan(self.pfw_enter)):
            if (self.lpfw_enter+self.starttime_cold-self.starttime_sync < 30):
                ax.axvline(x=self.lpfw_enter+self.starttime_cold-self.starttime_sync, color='b', linestyle='--', label='less_preferred_well_enter_time')
            if (self.lpfw_leave+self.starttime_cold-self.starttime_sync < 30):
                ax.axvline(x=self.lpfw_leave+self.starttime_cold-self.starttime_sync, color='purple', linestyle='--', label='less_preferred_well_leave_time')
        ax.legend(loc='upper right')
        fig.show()
        
        return
                

class cold_file:
    def __init__ (self,cold_folder,sync_folder,atlas_folder,input_format_df,day):
        
        self.day = day
        for filename in os.listdir(cold_folder):
            cold_day = int(re.findall(r'\d+', filename.split(input_format_df['day_tag'])[1])[0])
            if cold_day == self.day:
                self.df = pd.read_excel(os.path.join(cold_folder,filename))
                break
        for filename in os.listdir(sync_folder):
            #in this case I write down trails with an Atlas recording as the filename of an empty docx document
            if filename.endswith(input_format_df['key_suffix']):
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
                if input_format_df['sync_tag'] in filename:
                    ID = re.findall(r'\d+', filename.split(input_format_df['sync_tag'])[1])[0]
                    if int(ID) == i:
                        sync_file = pd.read_csv(os.path.join(sync_folder, filename))
                        
            for foldername in os.listdir(atlas_folder):
                
                if input_format_df['atlas_folder_tag'] in foldername:
                    ID = re.findall(r'\d+', foldername.split(input_format_df['atlas_folder_tag'])[1])[0]
                    print(foldername,ID,index)
                    if int(ID) == index+1:
                        folder_path = os.path.join(atlas_folder, foldername)
                        atlas_file = pd.read_csv(os.path.join(folder_path, input_format_df['atlas_z_filename']),header=None)
            
            self.key_trails.append(key_trail(self.df.iloc[[i]].reset_index(drop=True), sync_file, atlas_file))

def ObtainDayMax (input_format_df, parent_folder):
    day_max = -1
    for filename in os.listdir(parent_folder):
        if input_format_df['day_tag'] in filename:
            day = re.findall(r'\d+', filename.split(input_format_df['day_tag'])[1])[0]     
            day = int(day)
            if (day>day_max):
                day_max = day
    return day_max

def ObtainPreferredWell (cold_folder):
    for cold_filename in os.listdir(cold_folder):
        cold = pd.read_excel(os.path.join(cold_folder,cold_filename))
        w1 = 0
        w2 = 0
        for j in range (cold.shape[0]):
            if (cold['firstwellreached'][j]==1):
                w1+=1
            elif (cold['firstwellreached'][j]==2):
                w2+=1
        global pfw
        if w1>w2:
            pfw = 1
        else: 
            pfw = 2
    return
    
def ReadInFiles (input_format_df,parent_folder):
    day_max = ObtainDayMax(input_format_df, parent_folder)
    cold_folder = None
    files = [[None for _ in range(2)] for _ in range(day_max)]
    for filename in os.listdir(parent_folder):
        if input_format_df['cold_folder_tag'] in filename:
            cold_folder = os.path.join(parent_folder,filename)
            ObtainPreferredWell(cold_folder)
        elif input_format_df['day_tag'] in filename:
            day = int(re.findall(r'\d+', filename.split(input_format_df['day_tag'])[1])[0])
            if input_format_df['atlas_parent_folder_tag'] in filename:
                files[day-1][1] = os.path.join(parent_folder,filename)
            elif input_format_df['bonsai_folder_tag'] in filename:
                files[day-1][0] = os.path.join(parent_folder,filename)
    cold_files = []
    for i in range (0,day_max):
        cold_files.append(cold_file(cold_folder,files[i][0],files[i][1],input_format_df,i+1))
    return

def MainFunction (input_format_df):
    return

# atlas_folder = 'E:\Mingshuai\Group D\1769568/'
# sync_folder = '/Users/zhumingshuai/Desktop/Programming/Data/Atlas/Sample/'
# cold_folder = '/Users/zhumingshuai/Desktop/Programming/Data/Atlas/Sample/Training_Data_Day1.xlsx'
# a = cold_file(cold_folder,sync_folder,atlas_folder,input_format_df)

parent_folder = '//cmvm.datastore.ed.ac.uk/cmvm/sbms/users/s2764793/Win7/Desktop/workingfolder/Group D/1819287/'
ReadInFiles(input_format_df, parent_folder)