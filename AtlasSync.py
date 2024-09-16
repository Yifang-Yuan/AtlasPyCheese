
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
import scipy.signal as signal
from scipy import stats
import seaborn

input_format_df = {
    'sync_tag':'sync',
    'track_tag': 'tracking',
    'bonsai_folder_tag':'Bonsai',
    'atlas_folder_tag':'Atlas_Trial',
    'atlas_parent_folder_tag':'Atlas',
    'key_suffix':'.docx',
    'atlas_z_filename':'Zscore_trace.csv',
    'atlas_green_filename':'Green_trace.csv',
    'cold_folder_tag':'Cold_folder',
    'day_tag': 'Day',
    'key_ignore_time':120,
    'atlas_frame_rate': 840,
    'bonsai_frame_rate': 16,
    'atlas_recording_time':30,    
    'before_win': 2,
    'after_win': 2,
    'low_pass_filter_frequency': 80,
    'parent_folder': 'E:/workingfolder/Group D/1819287/',
    'MouseID': '1819287',
    'output_folder': 'SingleTrailPlot'
    }

pfw = None
test = None

class key_trail:
    def __init__(self,cold,sync,track,atlas,green,day,trail_ID,input_format_df):
        self.trail_ID = trail_ID
        self.day = day
        self.cold = cold
        self.sync = sync
        self.track = track
        self.atlas = atlas
        self.green = green
        self.Synchronisation(input_format_df)
        self.smoothed_atlas = pd.DataFrame(af.smooth_signal(self.atlas[0],window_len=10),columns=[0])
        self.output_path = os.path.join(input_format_df['parent_folder'],input_format_df['output_folder'])
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.PlotSingleTrail(input_format_df)
        self.PlotRewardCollection(input_format_df)
        speed_path = os.path.join(input_format_df['parent_folder'],'speed_files')
        speed_path = os.path.join(speed_path,'Day'+str(self.day))
        if not os.path.exists(speed_path):
            os.makedirs(speed_path)
        speed_tot=self.CalculateInstantSpeed(input_format_df)
        if speed_tot is not None:
            speed_tot.to_csv(os.path.join(speed_path,'Green&Speed_Day'+str(self.day)+'-'+str(self.trail_ID)+'.csv'),index = True)
        
    def Synchronisation (self,input_format_df):
        for i in range (len(self.sync)):
            if np.isnan(self.sync['Value.X'].iloc[i]) and np.isnan(self.sync['Value.Y'].iloc[i]):
                self.startframe_sync = i
                self.starttime_sync = i/input_format_df['bonsai_frame_rate']
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
    
    def CalculateInstantSpeed (self,input_format_df,start_frame=0,end_frame=int(input_format_df['atlas_recording_time']*input_format_df['atlas_frame_rate'])):
        raw_atlas = []
        fil_atlas = []
        raw_green = []
        fil_green = []
        speed = []
        bonsai_start_frame = int(round(start_frame/input_format_df['atlas_frame_rate']*input_format_df['bonsai_frame_rate']))
        bonsai_end_frame = int(round(end_frame/input_format_df['atlas_frame_rate']*input_format_df['bonsai_frame_rate']))
        self.filtered_green = LowPassFilter(self.green, input_format_df)
        if (self.sync.shape[0]<self.startframe_sync+int(input_format_df['atlas_recording_time']*input_format_df['bonsai_frame_rate'])-1):
            print('Bonsai recording does not fully cover atlas recording:')
            print('Day'+str(self.day)+' trail'+str(self.trail_ID))
            return
        for i in range (bonsai_start_frame,bonsai_end_frame):
            if (i==bonsai_end_frame-1):
                x = self.track['X'][i+self.startframe_sync]-self.track['X'][i+self.startframe_sync-1]
                y = self.track['Y'][i+self.startframe_sync]-self.track['Y'][i+self.startframe_sync-1]
            else:
                x = self.track['X'][i+self.startframe_sync+1]-self.track['X'][i+self.startframe_sync]
                y = self.track['Y'][i+self.startframe_sync+1]-self.track['Y'][i+self.startframe_sync]
            v = np.sqrt(pow(x,2)+pow(y,2))
            for j in range (int(i/input_format_df['bonsai_frame_rate']*input_format_df['atlas_frame_rate']),int(i/input_format_df['bonsai_frame_rate']*input_format_df['atlas_frame_rate']+input_format_df['atlas_frame_rate']/input_format_df['bonsai_frame_rate'])):
                atlas_frame = int(i/input_format_df['bonsai_frame_rate']*input_format_df['atlas_frame_rate']+j)
                speed.append(v)
                raw_atlas.append(self.atlas[0][j])
                fil_atlas.append(self.filtered_atlas[j][0])
                raw_green.append(self.atlas[0][j])
                fil_green.append(self.filtered_green[j][0])

        data = {
            'raw_z_score':raw_atlas,
            'raw_green':raw_green,
            'filtered_z_score':fil_atlas,
            'filtered_green':fil_green,
            'instant_speed':speed
            }
        GAS = pd.DataFrame(data)
        return GAS
      
    def PlotSingleTrail(self,input_format_df):
        title1 = 'Day'+str(self.day)+'_trail'+str(self.trail_ID)
        fig1, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.smoothed_atlas.index/input_format_df['atlas_frame_rate'],self.smoothed_atlas[0])
        ax.set_title(self.cold.loc[0,'name'])
        if not (np.isnan(self.pfw_enter)):
            if (self.pfw_enter+self.starttime_cold-self.starttime_sync < input_format_df['atlas_recording_time']):
                ax.axvline(x=self.pfw_enter+self.starttime_cold-self.starttime_sync, color='r', linestyle='--', label='preferred_well_enter_time')
                ax.legend(loc='upper right')
            if (self.pfw_leave+self.starttime_cold-self.starttime_sync < input_format_df['atlas_recording_time']):
                ax.axvline(x=self.pfw_leave+self.starttime_cold-self.starttime_sync, color='g', linestyle='--', label='preferred_well_leave_time')
                ax.legend(loc='upper right')
        if not (np.isnan(self.pfw_enter)):
            if (self.lpfw_enter+self.starttime_cold-self.starttime_sync < input_format_df['atlas_recording_time']):
                ax.axvline(x=self.lpfw_enter+self.starttime_cold-self.starttime_sync, color='b', linestyle='--', label='less_preferred_well_enter_time')
                ax.legend(loc='upper right')
            if (self.lpfw_leave+self.starttime_cold-self.starttime_sync < input_format_df['atlas_recording_time']):
                ax.axvline(x=self.lpfw_leave+self.starttime_cold-self.starttime_sync, color='purple', linestyle='--', label='less_preferred_well_leave_time')
                ax.legend(loc='upper right')
        ax.set_xlabel('time/s')
        ax.set_ylabel('z-score')
        fig1.savefig(os.path.join(self.output_path,title1+'.png'))
        plt.close(fig1)
        return
           
    def PlotRewardCollection(self,input_format_df):
        self.filtered_atlas = LowPassFilter(self.atlas, input_format_df)
        self.cropped_filtered_atlas = None
        self.cropped_filtered_atlas_lpfw = None
        target_len = (input_format_df['before_win']+input_format_df['after_win'])*input_format_df['atlas_frame_rate']
        time = np.arange(0,target_len)/input_format_df['atlas_frame_rate']-input_format_df['before_win']
        if not (np.isnan(self.pfw_enter)):
           pfw_enter_adj = self.pfw_enter+self.starttime_cold-self.starttime_sync
           
           #find out whether the collection is fully recorded or not
           if (pfw_enter_adj<(input_format_df['atlas_recording_time']-input_format_df['after_win'])) and (pfw_enter_adj>input_format_df['before_win']):
               title1 = 'Day'+str(self.day)+'_trail'+str(self.trail_ID)
               fig1, ax = plt.subplots(figsize=(10, 5))
               before_frame = int(round((pfw_enter_adj-input_format_df['before_win'])*input_format_df['atlas_frame_rate']))
               after_frame = int(round((pfw_enter_adj+input_format_df['after_win'])*input_format_df['atlas_frame_rate']))
               self.cropped_filtered_atlas = self.filtered_atlas[before_frame:after_frame]
               
               ax.plot(time,self.cropped_filtered_atlas,color='purple')
               ax.axvline(x=0,color='r', linestyle='--', label='reward collection at preferred well')
               ax.set_title(self.cold.loc[0,'name'])
               ax.set_xlabel('time/s')
               ax.set_ylabel('z-score')
               ax.legend(loc='upper right')
               fig1.savefig(os.path.join(self.output_path,title1+'_pfw.png'))
               plt.close(fig1)
               
               #the low band pass filter may accidentally change the length of array a bit 
               if len(self.cropped_filtered_atlas)>target_len:
                   self.cropped_filtered_atlas = self.cropped_filtered_atlas[:target_len]
               elif len(self.cropped_filtered_atlas)>target_len:
                   while(len(self.cropped_filtered_atlas)<target_len):
                       self.cropped_filtered_atlas = np.append(self.cropped_filtered_atlas,self.cropped_filtered_atlas[-1])
               
               #save speed files
               speed_path = os.path.join(input_format_df['parent_folder'],'speed_files')
               speed_path = os.path.join(speed_path,'Day'+str(self.day))
               
               if not os.path.exists(speed_path):
                   os.makedirs(speed_path)
               speed_file=self.CalculateInstantSpeed(input_format_df,start_frame=before_frame,end_frame=after_frame)
               if speed_file is not None:
                   speed_file=speed_file.set_index((time*input_format_df['atlas_frame_rate']).astype(int))
                   speed_file.to_csv(os.path.join(speed_path,'Speed_pfw_Day'+str(self.day)+'-'+str(self.trail_ID)+'.csv'),index = True)
               
        if not (np.isnan(self.lpfw_enter)):
           lpfw_enter_adj = self.lpfw_enter+self.starttime_cold-self.starttime_sync
           #find out whether the collection is fully recorded or not
           if (lpfw_enter_adj<(input_format_df['atlas_recording_time']-input_format_df['after_win'])) and (lpfw_enter_adj>input_format_df['before_win']):
               title1 = 'Day'+str(self.day)+'_trail'+str(self.trail_ID)
               fig1, ax = plt.subplots(figsize=(10, 5))
               before_frame = int(round((lpfw_enter_adj-input_format_df['before_win'])*input_format_df['atlas_frame_rate']))
               after_frame = int(round((lpfw_enter_adj+input_format_df['after_win'])*input_format_df['atlas_frame_rate']))
               self.cropped_filtered_atlas_lpfw = self.filtered_atlas[before_frame:after_frame]
               ax.plot(np.arange(0,len(self.cropped_filtered_atlas_lpfw))/input_format_df['atlas_frame_rate']-input_format_df['before_win'],self.cropped_filtered_atlas_lpfw,color='green')
               ax.axvline(x=0,color='r', linestyle='--', label='reward collection at less preferred well')
               ax.set_title(self.cold.loc[0,'name'])
               ax.set_xlabel('time/s')
               ax.set_ylabel('z-score')
               ax.legend(loc='upper right')
               fig1.savefig(os.path.join(self.output_path,title1+'_lpfw.png'))
               plt.close(fig1)
               
               #the low band pass filter may accidentally change the length of array a bit
               if len(self.cropped_filtered_atlas_lpfw)>target_len:
                   self.cropped_filtered_atlas_lpfw = self.cropped_filtered_atlas_lpfw[:target_len]
               elif len(self.cropped_filtered_atlas_lpfw)>target_len:
                   while(len(self.cropped_filtered_atlas_lpfw)<target_len):
                       self.cropped_filtered_atlas_lpfw = np.append(self.cropped_filtered_atlas_lpfw,self.cropped_filtered_atlas_lpfw[-1])
                
               #save speed files
               speed_path = os.path.join(input_format_df['parent_folder'],'speed_files')
               speed_path = os.path.join(speed_path,'Day'+str(self.day))
               if not os.path.exists(speed_path):
                   os.makedirs(speed_path)
               speed_file=self.CalculateInstantSpeed(input_format_df,start_frame=before_frame,end_frame=after_frame)
               if speed_file is not None:
                   speed_file=speed_file.set_index((time*input_format_df['atlas_frame_rate']).astype(int))
                   speed_file.to_csv(os.path.join(speed_path,'Speed_lpfw_Day'+str(self.day)+'-'+str(self.trail_ID)+'.csv'),index = True)
        
        #Obtain a random time for shuffle analysis
        lower_bound = input_format_df['before_win']
        upper_bound = input_format_df['atlas_recording_time']-input_format_df['after_win']
        self.ran_centre = np.random.uniform(low=lower_bound, high=upper_bound)
        before_frame = round((self.ran_centre-input_format_df['before_win'])*input_format_df['atlas_frame_rate'])
        after_frame = round((self.ran_centre+input_format_df['after_win'])*input_format_df['atlas_frame_rate'])
        self.random_atlas1= self.filtered_atlas[before_frame:after_frame]
        lower_bound = input_format_df['before_win']
        upper_bound = input_format_df['atlas_recording_time']-input_format_df['after_win']
        self.ran_centre = np.random.uniform(low=lower_bound, high=upper_bound)
        before_frame = round((self.ran_centre-(input_format_df['before_win']))*input_format_df['atlas_frame_rate'])
        after_frame = round((self.ran_centre+input_format_df['after_win'])*input_format_df['atlas_frame_rate'])
        self.random_atlas2= self.filtered_atlas[before_frame:after_frame]
        return
               
class cold_file:
    def __init__ (self,cold_folder,bonsai_folder,atlas_folder,input_format_df,day):
        self.day = day
        for filename in os.listdir(cold_folder):
            if not filename.endswith('.xlsx'):
                continue
            cold_day = int(re.findall(r'\d+', filename.split(input_format_df['day_tag'])[1])[0])
            if cold_day == self.day:
                self.df = pd.read_excel(os.path.join(cold_folder,filename))
                break
        legit = False
        for filename in os.listdir(bonsai_folder):
            #in this case I write down trails with an Atlas recording as the filename of an empty docx document
            if filename.endswith(input_format_df['key_suffix']):
                self.key_index = re.findall(r'\d+', filename)[0]
                legit = True
        if not legit:
            self.filtered_atlas_day = pd.DataFrame()
            return
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
        
        self.key_trails = []
        for index, i in enumerate(self.keynum):
            for filename in os.listdir(bonsai_folder):
                if input_format_df['sync_tag'] in filename:
                    ID = re.findall(r'\d+', filename.split(input_format_df['sync_tag'])[1])[0]
                    if int(ID) == i:
                        sync_file = pd.read_csv(os.path.join(bonsai_folder, filename))
                if input_format_df['track_tag'] in filename:
                    ID = re.findall(r'\d+', filename.split(input_format_df['track_tag'])[1])[0]
                    if int(ID) == i:
                        track_file = pd.read_csv(os.path.join(bonsai_folder, filename))       
            
            for foldername in os.listdir(atlas_folder):
                if input_format_df['atlas_folder_tag'] in foldername:
                    ID = re.findall(r'\d+', foldername.split(input_format_df['atlas_folder_tag'])[1])[0]
                    if int(ID) == index+1:
                        folder_path = os.path.join(atlas_folder, foldername)
                        atlas_file = pd.read_csv(os.path.join(folder_path, input_format_df['atlas_z_filename']),header=None)
                        green_file = pd.read_csv(os.path.join(folder_path, input_format_df['atlas_z_filename']),header=None)
            
            self.key_trails.append(key_trail(self.df.iloc[[i]].reset_index(drop=True), sync_file,track_file,atlas_file,green_file,self.day,i,input_format_df))
        self.IntegrateAtlas(input_format_df)
            
    def IntegrateAtlas(self,input_format_df):
        target_len = int((input_format_df['before_win']+input_format_df['after_win'])*input_format_df['atlas_frame_rate'])
        temp_atlas_day = np.empty((target_len,0))
        tempran_atlas_day = np.empty((target_len,0))
        D = []
        T = []
        
        for i in self.key_trails:
            if i.cropped_filtered_atlas is not None:
                global test
                test = i.random_atlas1
                temp_atlas_day = np.concatenate((temp_atlas_day, i.cropped_filtered_atlas), axis=1)
                D.append('Day'+str(self.day))
                T.append('Trail'+str(i.trail_ID))
                tempran_atlas_day = np.concatenate((tempran_atlas_day, i.random_atlas1), axis=1)
            if i.cropped_filtered_atlas_lpfw is not None:
                temp_atlas_day = np.concatenate((temp_atlas_day, i.cropped_filtered_atlas_lpfw), axis=1)
                D.append('Day'+str(self.day))
                T.append('Trail'+str(i.trail_ID))
                tempran_atlas_day = np.concatenate((tempran_atlas_day, i.random_atlas2), axis=1)
        
        header = pd.MultiIndex.from_arrays([D,T])
        if len(temp_atlas_day) == 0:
            return
        
        self.filtered_atlas_day = pd.DataFrame(temp_atlas_day,columns=header)
        self.filtered_atlas_day.columns.names = ['Day', 'Trail']
        
        mean = []
        CI = []
        
        for i in range (self.filtered_atlas_day.shape[0]):
            mean.append(self.filtered_atlas_day.iloc[i,:].mean())
            std = np.std(self.filtered_atlas_day.iloc[i,:])
            n = self.filtered_atlas_day.shape[1]
            t_value = stats.t.ppf(0.975, df=n-1)
            CI.append(t_value*std/np.sqrt(n))
        
        fig,ax = plt.subplots(figsize=(10, 5))
        time = np.arange(0,target_len)/input_format_df['atlas_frame_rate']-input_format_df['before_win']
        ax.plot(time, mean, color='blue', label='Mean Signal')
        CI = np.array(CI)
        mean = np.array(mean)
        ax.axvline(x=0,color='r', linestyle='--', label='reward collection')
        ax.fill_between(time, mean-CI, mean+CI, color='blue', alpha=0.2, label='95% CI')
        ax.set_xlabel('time/s')
        ax.set_ylabel('z-score')
        ax.set_title('Day'+str(self.day)+' average PETH')
        
        ran_mean = []
        ran_CI = []
        
        self.ran_atlas_day = pd.DataFrame(tempran_atlas_day,columns=header)
        self.filtered_atlas_day.columns.names = ['Day', 'Trail']
        
        for i in range (self.ran_atlas_day.shape[0]):
            ran_mean.append(self.ran_atlas_day.iloc[i,:].mean())
            std = np.std(self.ran_atlas_day.iloc[i,:])
            n = self.ran_atlas_day.shape[1]
            t_value = stats.t.ppf(0.975, df=n-1)
            ran_CI.append(t_value*std/np.sqrt(n))
        
        ran_mean = np.array(ran_mean)
        ran_CI = np.array(ran_CI)
        ax.plot(time, ran_mean, color='grey', label='Shuffle analysis')
        ax.legend(loc='upper right')
        ax.fill_between(time, ran_mean-ran_CI, ran_mean+ran_CI, color='grey', alpha=0.2, label='95% CI')
        
        output_path = os.path.join(input_format_df['parent_folder'],'PETH')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = os.path.join(output_path,'Day'+str(self.day)+'_avg_PETH.png')
        fig.savefig(output_path)
        
        return
    
def ObtainDayMax (input_format_df, parent_folder):
    day_max = -1
    for filename in os.listdir(parent_folder):
        if input_format_df['day_tag'] in filename:
            day = re.findall(r'\d+', filename.split(input_format_df['day_tag'])[1])[0]     
            day = int(day)
            if (day>day_max):
                day_max = day
    return day_max

def LowPassFilter (x,input_format_df):
    # Sampling frequency (in Hz)
    fs = input_format_df['atlas_frame_rate'] 
    cutoff = input_format_df['low_pass_filter_frequency']
    # Normalized cutoff frequency (cutoff frequency / Nyquist frequency)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Order of the filter
    order = 5
    # Get the filter coefficients
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y=signal.filtfilt(b, a, x, axis=0)
    return y
    
def ObtainPreferredWell (cold_folder):
    for cold_filename in os.listdir(cold_folder):
        if not cold_filename.endswith('.xlsx'):
            continue
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
   
def ReadInFiles (input_format_df):
    parent_folder = input_format_df['parent_folder']
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
    return cold_files

def PlotMousePETH (cold_files,input_format_df,mouse_ID):
    target_len = int((input_format_df['before_win']+input_format_df['after_win'])*input_format_df['atlas_frame_rate'])
    atlas_tot = np.empty((target_len, 0))
    ran_atlas_tot = np.empty((target_len, 0))
    for i in cold_files:
        if i.filtered_atlas_day.size!=0:
            atlas_tot = np.concatenate((atlas_tot, i.filtered_atlas_day), axis=1)
            ran_atlas_tot = np.concatenate((ran_atlas_tot, i.ran_atlas_day), axis=1)
   
   
    mean = []
    CI = []
    ran_mean = []
    ran_CI = []
    
    for i in range (atlas_tot.shape[0]):
        mean.append(atlas_tot[i,:].mean())
        std = np.std(atlas_tot[i,:])
        n = atlas_tot.shape[1]
        t_value = stats.t.ppf(0.975, df=n-1)
        CI.append(t_value*std/np.sqrt(n))
    fig,ax = plt.subplots(figsize=(10, 5))
    time = np.arange(0,target_len)/input_format_df['atlas_frame_rate']-input_format_df['before_win']
    CI = np.array(CI)
    mean = np.array(mean)
    ax.plot(time, mean, color='blue', label='Mean Signal')
    ax.fill_between(time, mean-CI, mean+CI, color='blue', alpha=0.2, label='95% CI')
    ax.axvline(x=0,color='r', linestyle='--', label='reward collection')
    ax.set_xlabel('time/s')
    ax.set_ylabel('z-score')
    ax.set_title('Mouse'+str(mouse_ID)+' average PETH')
    
    for i in range (ran_atlas_tot.shape[0]):
        ran_mean.append(ran_atlas_tot[i,:].mean())
        std = np.std(ran_atlas_tot[i:,])
        n = ran_atlas_tot.shape[1]
        t_value = stats.t.ppf(0.975, df=n-1)
        ran_CI.append(t_value*std/np.sqrt(n))
    
    ran_mean = np.array(ran_mean)
    ran_CI = np.array(ran_CI)
    ax.plot(time, ran_mean, color='grey', label='Shuffle analysis')
    ax.legend(loc='upper right')
    ax.fill_between(time, ran_mean-ran_CI, ran_mean+ran_CI, color='grey', alpha=0.2, label='95% CI')
    
    
    output_path = os.path.join(input_format_df['parent_folder'],'PETH')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path,'Mouse'+str(mouse_ID)+'_avg_PETH.png')
    
    fig.savefig(output_path)
    
def PlotMouseHeatMap (cold_files,input_format_df,mouse_ID):
    target_len = (input_format_df['before_win']+input_format_df['after_win'])*input_format_df['atlas_frame_rate']
    atlas_tot = pd.DataFrame()
    for i in cold_files:
        if i.filtered_atlas_day.size!=0:
            atlas_tot = pd.concat([atlas_tot,i.filtered_atlas_day],axis=1)
    
    time = np.arange(0,target_len)/input_format_df['atlas_frame_rate']-input_format_df['before_win']
    atlas_tot = atlas_tot.set_index(time)
    global test
    test = atlas_tot
    
    seaborn.heatmap(atlas_tot.T)
    plt.axvline(x=target_len/2,color='blue', linestyle='--')
    label = np.arange(-input_format_df['before_win'],input_format_df['after_win']+1,step=1)
    tick_positions = np.linspace(0, atlas_tot.shape[0]-1,len(label))
    
    plt.xticks(ticks = tick_positions,labels=label)
    output_path = os.path.join(input_format_df['parent_folder'],'PETH')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path,'Mouse'+str(mouse_ID)+'_heatmap.png')
    plt.savefig(output_path)
    plt.close()
    return 
    
    
def MainFunction (input_format_df,mouse_ID):
    cold_files = ReadInFiles(input_format_df)
    PlotMousePETH (cold_files,input_format_df,mouse_ID)
    PlotMouseHeatMap(cold_files, input_format_df, mouse_ID)
    return cold_files

# atlas_folder = 'E:\Mingshuai\Group D\1769568/'
# sync_folder = '/Users/zhumingshuai/Desktop/Programming/Data/Atlas/Sample/'
# cold_folder = '/Users/zhumingshuai/Desktop/Programming/Data/Atlas/Sample/Training_Data_Day1.xlsx'
# a = cold_file(cold_folder,sync_folder,atlas_folder,input_format_df)
 
cold_files=MainFunction(input_format_df,input_format_df['MouseID'])
