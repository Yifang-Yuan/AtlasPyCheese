# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:20:45 2024

@author: s2764793
"""
import os
import pandas as pd
import re
import numpy as np
import math

parameter = {
    'tracking_split_tag':'cam',
    'tracking_file_sufix': '.csv',
    'tracking_file_tag': 'DLC',
    'well_coord': [[473,127],[228,91]],
    'detecting_radius': 20,
    'point_of_no_return': 0.5,
    'CamFs':24,
    'sig_level':0.9
    }

class frame:
    def __init__(self,head,shoulder,bottom,parameter):
        self.head = head
        self.shoulder = shoulder
        self.bottom = bottom
        self.well1 = parameter['well_coord'][0]
        self.well2 = parameter['well_coord'][1]
        self.r = parameter['detecting_radius']
        self.ang = math.degrees(Ang(self.shoulder,self.head))
        print(self.ang)
    
    def IsCloseToWell(self):
        if Dis(self.well1,self.well2)<=self.r*2:
            print('Error,the two wells are too close!')
            return
        if Dis(self.head,self.well1)<=self.r:
            self.well = self.well1
            self.well_tag = 1
            # print('Close to well 1')
            return True
        if Dis(self.head,self.well2)<=self.r:
            self.well = self.well2
            self.well_tag = 2
            # print('Close to well 2')
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
            if not file.iloc[i,0].isdigit():
                continue
            for j in range (file.shape[1]):
                if (file.iloc[self.bdp_index,j] == 'head'):
                    if (file.iloc[self.xy_index,j]=='x'):
                        head_x = float(file.iloc[i,j])
                    if (file.iloc[self.xy_index,j]=='y'):
                        head_y = float(file.iloc[i,j])
                if (file.iloc[self.bdp_index,j] == 'shoulder'):
                    if (file.iloc[self.xy_index,j]=='x'):
                        shoulder_x = float(file.iloc[i,j])
                    if (file.iloc[self.xy_index,j]=='y'):
                        shoulder_y = float(file.iloc[i,j])
                if (file.iloc[self.bdp_index,j] == 'bottom'):
                    if (file.iloc[self.xy_index,j]=='x'):
                        bottom_x = float(file.iloc[i,j])
                    if (file.iloc[self.xy_index,j]=='y'):
                        bottom_y = float(file.iloc[i,j])
            single_frame = frame([head_x,head_y],[shoulder_x,shoulder_y],[bottom_x,bottom_y],parameter)
            self.frames.append(single_frame)
        self.Marking(parameter)
        self.SaveFile()
        print(self.BoarderForce)
        
            
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
                if self.BoarderPass(i,True):
                    current_state = 'Inside'
                    self.BoarderForce['Enter'].append(i/parameter['CamFs'])
                    self.BoarderForce['Well'].append(frame.well_tag)
            if (not (frame.IsCloseToWell()) and current_state == 'Inside'):
                if self.BoarderPass(i,False):
                    current_state = 'Outside'
                    self.BoarderForce['Leave'].append(i/parameter['CamFs'])
                    
    
    def BoarderPass (self,index,expectation):
        frame_of_no_return = round(parameter['point_of_no_return']*parameter['CamFs'])
        legit = 0
        outlaw = 0
        for i in self.frames[index:min(len(self.frames),index+frame_of_no_return)]:
            if (i.IsCloseToWell() == expectation):
                legit+=1
            else:
                outlaw+=1
        print(legit,outlaw)
        ratio = legit/(legit+outlaw)
        if ratio >= parameter['sig_level']:
            return True
        else:
            return False
        
    def SaveFile (self):
        data = {
            'head_x':[],
            'head_y':[],
            'shoulder_x':[],
            'shoulder_y':[],
            'bottom_x':[],
            'bottom_y':[],
            'angle':[]
            }
        for i in self.frames:
            data['head_x'].append(i.head[0])
            data['head_y'].append(i.head[1])
            data['shoulder_x'].append(i.shoulder[0])
            data['shoulder_y'].append(i.shoulder[1])
            data['bottom_x'].append(i.bottom[0])
            data['bottom_y'].append(i.bottom[1])
            data['angle'].append(i.ang)
        self.df = pd.DataFrame(data)
        
        
class dayx:
    def __init__(self,parameter,folder,day,op_path):
        self.day = day
        self.trails = []
        output_path = os.path.join(op_path,'Day'+str(day))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for filename in os.listdir(folder):
            if filename.endswith(parameter['tracking_file_sufix']) and parameter['tracking_file_tag'] in filename:
                file_path = os.path.join(folder,filename)
                ID = int(re.findall(r'\d+', filename.split(parameter['tracking_split_tag'])[1])[0])
                file = pd.read_csv(file_path)
                print(ID,day)
                t = trace(file,ID,self.day,parameter)
                df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in t.BoarderForce.items()]))
                print(output_path)
                df.to_csv(os.path.join(output_path,'Day'+str(day)+'-'+str(ID)+'_tracking.csv'))
                t.df.to_csv(os.path.join(output_path,'Day'+str(day)+'-'+str(ID)+'_frames.csv'))
                self.trails.append(t)   

class mice:
    def __init__(self,parameter,parent_folder,mouse_ID):
        self.mice_days = []
        output_path = os.path.join(parent_folder,'output')
        output_path = os.path.join(output_path,mouse_ID)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for filename in os.listdir(parent_folder):
            if 'day' in filename:
                day = int(re.findall(r'\d+', filename.split('day')[1])[0])
                folder = os.path.join(parent_folder,filename)
            elif 'Day' in filename:
                day = int(re.findall(r'\d+', filename.split('Day')[1])[0])
                folder = os.path.join(parent_folder,filename)
            elif 'Probe' in filename or 'probe' in filename:
                day = -1
                folder = os.path.join(parent_folder,filename)
            else:
                continue
            self.mice_days.append(dayx(parameter,folder,day,output_path))
          


def Dis (x,y):
    return np.sqrt((y[0]-x[0])**2+(y[1]-x[1])**2)

def Ang (x,y):
    return math.atan2((y[1]-x[1]),(y[0]-x[0]))
     
folder = '//cmvm.datastore.ed.ac.uk/cmvm/sbms/users/s2764793/Win7/Desktop/DLC/1054/'
a = mice(parameter,folder,'1054')

                
                