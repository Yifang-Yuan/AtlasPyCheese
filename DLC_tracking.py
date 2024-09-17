# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:20:45 2024

@author: s2764793
"""
import os
import pandas as pd
import re
import numpy as np

parameter = {
    'split_tag':'cam',
    'tracking_file_sufix': '.csv',
    'tracking_file_tag': 'DLC',
    'well_coord': [[30,240],[250,240]],
    'detecting_radius': 20
    }

class frame:
    def __init__(self,head,shoulder,bottom):
        self.head = head
        self.shoulder = shoulder
        self.bottom = bottom

class trace:
    def __init__(self,file,ID,day,parameter):
        self.file = file
        self.ID = ID
        self.day = day
        self.frames = []
        for i in range (file.shape[0]):
            if file.iloc[i,0] == 'bodyparts':
                self.bdp_index = i
            if file.iloc[i,0] == 'coords':
                self.xy_index = i
        for i in range (file.shape[0]):
            for j in range (file.shape[1]):
                if (file.iloc[self.bdp_index,j] == 'head'):
                    if (file.iloc[self.xy_index,j]=='x'):
                        head_x = file.iloc[i,j]
                    if (file.iloc[self.xy_index,j]=='y'):
                        head_y = file.iloc[i,j] 
                if (file.iloc[self.bdp_index,j] == 'shoulder'):
                    if (file.iloc[self.xy_index,j]=='x'):
                        shoulder_x = file.iloc[i,j]
                    if (file.iloc[self.xy_index,j]=='y'):
                        shoulder_y = file.iloc[i,j] 
                if (file.iloc[self.bdp_index,j] == 'bottom'):
                    if (file.iloc[self.xy_index,j]=='x'):
                        bottom_x = file.iloc[i,j]
                    if (file.iloc[self.xy_index,j]=='y'):
                        bottom_y = file.iloc[i,j] 
            single_frame = frame([head_x,head_y],[shoulder_x,shoulder_y],[bottom_x,bottom_y])
            self.frames.append(single_frame)
        
    def MouseRadar (parameter):
        #distance to well 1
        well1_coord = parameter['well_coord'][0]
        well2_coord = parameter['well_coord'][1]
        
        return
        
class dayx:
    def __init__(self,parameter,folder,day):
        self.day = day
        for filename in os.listdir(folder):
            if filename.endswith(parameter['tracking_file_sufix']) and parameter['tracking_file_tag'] in filename:
                file_path = os.path.join(folder,filename)
                ID = int(re.findall(r'\d+', filename.split(parameter['split_tag'])[1])[0])
                file = pd.read_csv(file_path)
                print(ID,day)
                t = trace(file,ID,self.day,parameter)
                
def Dis (x,y):
    return np.sqrt((y[0]-x[0])**2+(y[1]-x[1])**2)
     
folder = '//cmvm.datastore.ed.ac.uk/cmvm/sbms/users/s2764793/Win7/Desktop/DLC/'
a = dayx(parameter,folder,1)

                
                