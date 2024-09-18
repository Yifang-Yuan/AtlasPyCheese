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
    'detecting_radius': 10,
    'point_of_no_return': 1,
    'CamFs':16,
    'sig_level':0.95
    }

class frame:
    def __init__(self,head,shoulder,bottom,parameter):
        self.head = head
        self.shoulder = shoulder
        self.bottom = bottom
        self.well1 = parameter['well_coord'][0]
        self.well2 = parameter['well_coord'][1]
        self.r = parameter['detecting_radius']
    
    def IsCloseToWell(self):
        if Dis(self.well1,self.well2)<=self.r:
            print('Error,the two wells are too close!')
            return
        if Dis(self.head,self.well1)<=self.r:
            self.well = self.well1
            self.well_tag = 1
            return True
        if Dis(self.head,self.well2)<=self.r:
            self.well = self.well2
            self.well_tag = 2
            return True
        return False
    
    def IsInward(self):
        if Dis(self.head,self.well)<=Dis(self.shoulder,self.well):
            return True
        else:
            return False

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
            single_frame = frame([head_x,head_y],[shoulder_x,shoulder_y],[bottom_x,bottom_y],parameter)
            self.frames.append(single_frame)
        
    def Marking (self,parameter):
        current_state = 'Outside'
        self.BoarderForce = {
            'Enter':[],
            'Leave':[],
            'Well':[]
            }
        
        for i in range (len(self.frames)):
            frame = self.frames[i]
            #This refers that the mouse may trying to reach the well
            if (frame.IsCloseToWell() and current_state == 'Outside'):
                if self.BoarderPass(parameter,i,True):
                    current_state = 'Inside'
                    self.BoarderForce['Leave'].append(i/parameter['CamFs'])
                    self.BoarderForce['well'].append(frame.well_tag)
            if (not (frame.IsCloseToWell()) and current_state == 'Inside'):
                if self.BoarderPass(parameter,i,False):
                    current_state = 'Outside'
                    self.BoarderForce['Leave'].append(i/parameter['CamFs'])
                    
    
    def BoarderPass (self,parameter,index,expectation):
        frame_of_no_return = round(parameter['point_of_no_return']*parameter['CamFs'])
        legit = 0
        outlaw = 0
        for i in self.frames[index:max(len(self.frame),index+frame_of_no_return)]:
            if (self.frames[i].IsCloseToWell() == expectation):
                legit+=1
            else:
                outlaw+=1
        ratio = legit/(legit+outlaw)
        if ratio >= parameter['sig_level']:
            return True
        else:
            return False
        
        
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

                
                