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
import matplotlib.pyplot as plt

parameter = {
    'tracking_split_tag':'cam',
    'tracking_file_sufix': '.csv',
    'tracking_file_tag': 'DLC',
    'well_coord': [[498,211],[292,71]],
    'bridge_coord': [[37,216],[258,106]],
    'detecting_radius': 15,
    'point_of_no_return': 3,
    'CamFs':24,
    'sig_level':0.95
    }

class frame:
    def __init__(self,head,shoulder,bottom):
        self.head = head
        self.shoulder = shoulder
        self.bottom = bottom
        self.well1 = parameter['well_coord'][0]
        self.well2 = parameter['well_coord'][1]
        self.r = parameter['detecting_radius']
        self.ang = math.degrees(Ang(self.shoulder,self.head))
    
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
    
    def IsOnBridge(self):
        x1 = min(parameter['bridge_coord'][0][0],parameter['bridge_coord'][1][0])
        x2 = max(parameter['bridge_coord'][0][0],parameter['bridge_coord'][1][0])
        y1 = min(parameter['bridge_coord'][0][1],parameter['bridge_coord'][1][1])
        y2 = max(parameter['bridge_coord'][0][1],parameter['bridge_coord'][1][1])
        if (self.head[0]>=x1 and self.head[0]<=x2) and (self.head[1]>=y1 and self.head[1]<=y2):
            return True
        elif not ((self.bottom[0]>=x1 and self.bottom[0]<=x2) and (self.bottom[1]>=y1 and self.bottom[1]<=y2)):
            return False

class trace:
    def __init__(self,file,ID,day):
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
            single_frame = frame([head_x,head_y],[shoulder_x,shoulder_y],[bottom_x,bottom_y])
            self.frames.append(single_frame)
        # x_values = [frame.shoulder[0] for frame in self.frames]
        # y_values = [frame.shoulder[1] for frame in self.frames]
        
        self.Marking()
        plt.legend(loc='upper left')
        # Invert the axes to place (0, 0) in the upper-right corner
        plt.gca().invert_yaxis()  # Invert the y-axis (top to bottom)
        circle1 = plt.Circle(parameter['well_coord'][0], parameter['detecting_radius'], color='b', fill=True)
        circle2 = plt.Circle(parameter['well_coord'][1], parameter['detecting_radius'], color='b', fill=True)
        # Add the circles to the current plot
        plt.gca().add_patch(circle1)
        plt.gca().add_patch(circle2)
        # Equal scaling for both axes
        plt.gca().set_aspect('equal')
        plt.show()
        self.SaveFile()
            
    def Marking (self,):
        frame_stamp = 0
        current_state = 'Outside'
        self.Bridge_detection = 'off'
        self.enter_CB = None
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
                    x,y = self.ObtainShoulderCoord(frame_stamp,i)
                    t1 = round(frame_stamp/parameter['CamFs'])
                    t2 = round(i/parameter['CamFs'])
                    lab = 'From'+str(t1)+'To'+str(t2)+'(s)'
                    plt.plot(x,y,label = lab)
                    frame_stamp = i
            if (not (frame.IsCloseToWell()) and current_state == 'Inside'):
                if self.BoarderPass(i,False):
                    current_state = 'Outside'
                    self.BoarderForce['Leave'].append(i/parameter['CamFs'])
                    x,y = self.ObtainShoulderCoord(frame_stamp,i)
                    t1 = round(frame_stamp/parameter['CamFs'])
                    t2 = round(i/parameter['CamFs'])
                    lab = 'From'+str(t1)+'To'+str(t2)+'(s)'
                    # plt.plot(x,y,label = lab)
                    frame_stamp = i
            if (self.Bridge_detection == 'off' and frame.IsOnBridge()):
                self.Bridge_detection = 'on'
                self.enter_CB = i/parameter['CamFs']
        x,y = self.ObtainShoulderCoord(frame_stamp,len(self.frames))
        print(len(x),len(y))
        t1 = round(frame_stamp/parameter['CamFs'])
        t2 = round(i/parameter['CamFs'])
        lab = 'From'+str(t1)+'To'+str(t2)+'(s)'
        plt.plot(x,y,label = lab)
        frame_stamp = i
    
    def BoarderPass (self,index,expectation):
        
        frame_of_no_return = round(parameter['point_of_no_return']*parameter['CamFs'])
        legit = 0
        outlaw = 0
        for i in self.frames[index:min(len(self.frames),index+frame_of_no_return)]:
            if (i.IsCloseToWell() == expectation):
                legit+=1
            else:
                outlaw+=1
        ratio = legit/(legit+outlaw)
        if ratio >= parameter['sig_level']:
            
            return True
        else:
            return False
    
    def ObtainShoulderCoord(self,f1,f2):
        x_values = [frame.shoulder[0] for frame in self.frames[f1:f2]]
        y_values = [frame.shoulder[1] for frame in self.frames[f1:f2]]
        return x_values,y_values        
        
    def SaveFile (self):
        data = {
            'head_x':[],
            'head_y':[],
            'shoulder_x':[],
            'shoulder_y':[],
            'bottom_x':[],
            'bottom_y':[],
            'angle':[],
            'speed':[]
            }
        for index,i in enumerate(self.frames):
            data['head_x'].append(i.head[0])
            data['head_y'].append(i.head[1])
            data['shoulder_x'].append(i.shoulder[0])
            data['shoulder_y'].append(i.shoulder[1])
            data['bottom_x'].append(i.bottom[0])
            data['bottom_y'].append(i.bottom[1])
            data['angle'].append(i.ang)
            if i != self.frames[-1]:
                data['speed'].append(Dis(self.frames[index+1].bottom,self.frames[index].bottom))
            else:
                data['speed'].append(data['speed'][-1])
                
        self.df = pd.DataFrame(data)
        
        
class dayx:
    def __init__(self,folder,day,op_path):
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
                print('Day'+str(self.day)+' Trial'+str(ID))
                t = trace(file,ID,self.day)
                df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in t.BoarderForce.items()]))
                df.to_csv(os.path.join(output_path,'Day'+str(day)+'-'+str(ID)+'_tracking.csv'))
                t.df.to_csv(os.path.join(output_path,'Day'+str(day)+'-'+str(ID)+'_frames.csv'))
                self.trails.append(t)  

class mice:
    def __init__(self,parent_folder,mouse_ID):
        self.mice_days = []
        output_path = os.path.join(parent_folder,'output')
        output_path = os.path.join(output_path,mouse_ID)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print('Now reading:'+str(mouse_ID))
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
            self.mice_days.append(dayx(folder,day,output_path))
            self.Neo_Cold(parent_folder)
     
    def Neo_Cold(self,folder):
        cold = {
            'Day':[],
            'Trail_ID':[],
            'Time_of_entry':[]
            }
        output_path = os.path.join(folder,'Neo_Cold.csv')
        print(output_path)
        for day in self.mice_days:
            for trail in day.trails:
                cold['Day'].append(day.day)
                cold['Trail_ID'].append(trail.ID)
                cold['Time_of_entry'].append(trail.enter_CB)
        df = pd.DataFrame(cold)
        df.to_csv(output_path)
     


def Dis (x,y):
    return np.sqrt((y[0]-x[0])**2+(y[1]-x[1])**2)

def Ang (x,y):
    return math.atan2((y[1]-x[1]),(y[0]-x[0]))
     
folder = '/Users/zhumingshuai/Desktop/Sample Data/1054/'
a = mice(folder,'1054')

                
                