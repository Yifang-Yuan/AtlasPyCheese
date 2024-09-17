# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:14:10 2024

@author: Yifang
"""

import pandas as pd
from waveletFunctions import wavelet
import os
import numpy as np
import matplotlib.pylab as plt
from scipy.signal import filtfilt
from scipy import signal
import numpy as np
import seaborn as sns
import photometry_functions as fp
import glob

def butter_filter(data, btype='low', cutoff=10, fs=9938.4, order=5): 
    # cutoff and fs in Hz
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
    y = signal.filtfilt(b, a, data, axis=0)
    return y

def band_pass_filter(data,low_freq,high_freq,Fs):
    data_high=butter_filter(data, btype='high', cutoff=low_freq,fs=Fs, order=5)
    data_low=butter_filter(data_high, btype='low', cutoff=high_freq, fs=Fs, order=5)
    return data_low

def notchfilter (data,f0=100,bw=10,fs=840):
    # Bandwidth of the notch filter (in Hz)   
    Q = f0/bw # Quality factor
    b, a = signal.iirnotch(f0, Q, fs)
    for _ in range(4):
        data = signal.filtfilt(b, a, data)
    return data
def Calculate_wavelet(signal_pd,lowpassCutoff=1500,Fs=10000,scale=40):
    if isinstance(signal_pd, np.ndarray)==False:
        signal=signal_pd.to_numpy()
    else:
        signal=signal_pd
    sst = butter_filter(signal, btype='low', cutoff=lowpassCutoff, fs=Fs, order=5)
    sst = sst - np.mean(sst)
    variance = np.std(sst, ddof=1) ** 2
    #print("variance = ", variance)
    # ----------C-O-M-P-U-T-A-T-I-O-N------S-T-A-R-T-S------H-E-R-E---------------
    if 0:
        variance = 1.0
        sst = sst / np.std(sst, ddof=1)
    n = len(sst)
    dt = 1/Fs

    pad = 1  # pad the time series with zeroes (recommended)
    dj = 0.25  # this will do 4 sub-octaves per octave
    s0 = scale * dt  # this says start at a scale of 10ms, use shorter scale will give you wavelet at high frequecny
    j1 = 7 / dj  # this says do 7 powers-of-two with dj sub-octaves each
    lag1 = 0.1  # lag-1 autocorrelation for red noise background
    #print("lag1 = ", lag1)
    mother = 'MORLET'
    # Wavelet transform:
    wave, period, scale, coi = wavelet(sst, dt, pad, dj, s0, j1, mother)
    power = (np.abs(wave)) ** 2  # compute wavelet power spectrum
    global_ws = (np.sum(power, axis=1) / n)  # time-average over all times
    frequency=1/period
    return sst,frequency,power,global_ws

def plot_wavelet(ax,sst,frequency,power,Fs=10000,colorBar=False,logbase=False):
    import matplotlib.ticker as ticker
    time = np.arange(len(sst)) /Fs   # construct time array
    level=8 #level is how many contour levels you want
    CS = ax.contourf(time, frequency, power, level)
    #ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')
    #ax.set_title('Wavelet Power Spectrum')
    #ax.set_xlim(xlim[:])
    if logbase:
        ax.set_yscale('log', base=2, subs=None)
    ax.set_ylim([np.min(frequency), np.max(frequency)])
    yax = plt.gca().yaxis
    yax.set_major_formatter(ticker.ScalarFormatter())
    if colorBar: 
        fig = plt.gcf()  # Get the current figure
        position = fig.add_axes([0.6, -0.01, 0.3, 0.02])
        #position = fig.add_axes()
        cbar=plt.colorbar(CS, cax=position, orientation='horizontal', fraction=0.05, pad=0.5)
        cbar.set_label('Power (mV$^2$)', fontsize=12) 
        #plt.subplots_adjust(right=0.7, top=0.9)              
    return -1

def plot_average_zscore_speed (dpath,Fs=840):
    pattern = os.path.join(dpath, 'Day*/', 'Speed_*.csv')
    #pattern = os.path.join(dpath, 'Day*/', 'Green&Speed_*.csv')
    # Get a list of all matching files
    file_list = glob.glob(pattern)
    # Loop through the file list and read each file
    dfs_speed = []
    dfs_zscore= []
    dfs_power=[]
    dfs_sst=[]
    dfs_power_ripple=[]
    dfs_sst_ripple=[]
    dfs=[]
    frequency=[]
    frequency_ripple=[]
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)
            df['instant_speed'] = df['instant_speed'].mask(df['instant_speed'] > 16)
            df['instant_speed'].interpolate(method='nearest', inplace=True)
            df['instant_speed'].fillna(method='bfill', inplace=True)  # Fill NaNs at the beginning
            df['instant_speed'].fillna(method='ffill', inplace=True)  # Fill NaNs at the end
            df.loc[df['instant_speed'] > 10, 'instant_speed'] = 10
            speed_data=df['instant_speed'].values
            speed_data=butter_filter(speed_data, btype='low', cutoff=20, fs=Fs, order=5)
            reshaped_speed = speed_data.reshape(1, -1)
            
            zscore_raw = df['raw_z_score'].values
            zscore_raw=notchfilter (zscore_raw,f0=100,bw=10,fs=Fs)
            zscore_smooth=fp.smooth_signal(zscore_raw,window_len=8,window='flat')
            reshaped_zscore = zscore_raw.reshape(1, -1)
            
            zscore_bandpass = band_pass_filter(zscore_smooth, 5, 60, Fs)
            zscore_theta_bandpass = band_pass_filter(zscore_smooth, 4, 15, Fs)
            #reshaped_zscore_bandpass=zscore_bandpass.reshape(1, -1)
            sst, frequency, power, _ = Calculate_wavelet(zscore_bandpass, lowpassCutoff=100, Fs=Fs, scale=10)
           
            zscore_ripple_bandpass=band_pass_filter(zscore_raw,130,180,Fs)
            sst_ripple,frequency_ripple,power_ripple,_=Calculate_wavelet(zscore_ripple_bandpass,lowpassCutoff=180,Fs=Fs,scale=2)     

            time_axis = np.arange(len(zscore_bandpass)) /Fs # Create a time axis
            
            #fig, ax = plt.subplots(6, 1, figsize=(8, 10),gridspec_kw={'height_ratios': [1, 1,2,2, 3,3]})
            fig, ax = plt.subplots(6, 1, figsize=(10, 10))
            heatmap =sns.heatmap(reshaped_speed, cmap='magma', annot=False, cbar=False, ax=ax[0])
            ax[0].set_title("Heatmap of Speed Data")
            ax[0].tick_params(labelbottom=False)  # Remove x-tick labels
            ax[0].tick_params(bottom=False)  # Remove x-ticks
    
            heatmap_zscore =sns.heatmap(reshaped_zscore, cmap='magma', annot=False, cbar=False, ax=ax[1])
            ax[1].set_title("Heatmap of Zscore")
            ax[1].tick_params(labelbottom=False)  # Remove x-tick labels
            ax[1].tick_params(bottom=False)  # Remove x-ticks
            
            ax[2].plot(time_axis, zscore_smooth, color='k', label='Smoothed Z-Score')
            ax[3].plot(time_axis, zscore_theta_bandpass, color='blue', label='Theta band Z-Score')
            for axis in [ax[2], ax[3]]:
                # Remove all spines
                axis.spines['top'].set_visible(False)
                axis.spines['right'].set_visible(False)
                axis.spines['left'].set_visible(False)
                axis.spines['bottom'].set_visible(False)
                axis.margins(0)
                axis.tick_params(left=False, bottom=False)  # Remove ticks on both axes
                axis.legend(loc='upper right', frameon=False)
                
            plot_wavelet(ax[4], sst, frequency, power, Fs, colorBar=False, logbase=True)
            ax[4].set_title("Theta band")
            
            plot_wavelet(ax[5], sst_ripple, frequency_ripple, power_ripple, Fs, colorBar=False, logbase=True)
            ax[5].set_title("Ripple band")
            
            # cbar_ax = fig.add_axes([ax[0].get_position().x0, ax[5].get_position().y0 - 0.05, 
            #                         ax[0].get_position().width*0.2, 0.02]) 
            # plt.colorbar(heatmap.collections[0], cax=cbar_ax, orientation='horizontal')
    
            # cbar_ax = fig.add_axes([ax[1].get_position().x0+0.2, ax[5].get_position().y0 - 0.05, 
            #                         ax[1].get_position().width*0.2, 0.02])  
            # plt.colorbar(heatmap_zscore.collections[0], cax=cbar_ax, orientation='horizontal')
            plt.tight_layout()
            plt.show()
            dfs.append(df)
            
            dfs_speed.append(df['instant_speed'].values)
            dfs_zscore.append (zscore_smooth)
            dfs_power.append(power)
            dfs_sst.append(sst)
            dfs_power_ripple.append(power_ripple)
            dfs_sst_ripple.append(sst_ripple)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return dfs_speed,dfs_zscore,dfs_power,dfs_sst,dfs_power_ripple,dfs_sst_ripple,frequency,frequency_ripple

def plot_heatmap_all(dpath,Fs):
    dfs_speed,dfs_zscore,dfs_power,dfs_sst,dfs_power_ripple,dfs_sst_ripple,frequency,frequency_ripple=plot_average_zscore_speed (dpath,Fs)
    # Convert lists to NumPy arrays
    dfs_speed = np.array(dfs_speed)  # Shape: (num_files, num_points)
    dfs_zscore = np.array(dfs_zscore)  # Shape: (num_files, num_points)
    dfs_power = np.array(dfs_power)  # Shape: (num_files, num_frequencies, num_points)
    dfs_sst=np.array(dfs_sst)
    dfs_power_ripple = np.array(dfs_power_ripple)  # Shape: (num_files, num_frequencies, num_points)
    dfs_sst_ripple=np.array(dfs_sst_ripple)
    # Compute the averages
    avg_speed = np.mean(dfs_speed, axis=0)
    avg_zscore = np.mean(dfs_zscore, axis=0)
    avg_power = np.mean(dfs_power, axis=0)
    avg_sst = np.mean(dfs_sst, axis=0)
    avg_power_ripple = np.mean(dfs_power_ripple, axis=0)
    avg_sst_ripple = np.mean(dfs_sst_ripple, axis=0)
    # Reshape the average arrays for heatmap (1, -1)
    reshaped_avg_speed = avg_speed.reshape(1, -1)
    reshaped_avg_zscore = avg_zscore.reshape(1, -1)
    
    sem_zscore = np.std(dfs_zscore, axis=0) / np.sqrt(dfs_zscore.shape[0])  # SEM
    ci95_zscore = 1.96 * sem_zscore  # 95% confidence interval

    #plot all heatmap
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    heatmap = sns.heatmap(dfs_speed, cmap='magma', annot=False, cbar=True, ax=ax)
    ax.set_title("Heatmap of Speed Data for Trials")
    
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    heatmap = sns.heatmap(dfs_zscore, cmap='magma', annot=False, cbar=True, ax=ax)
    ax.set_title("Heatmap of zscore Data for Trials")
    
    # Plot average heatmap
    fig, ax = plt.subplots(4, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [1, 1, 4, 2]})
    # Plot the average speed heatmap
    heatmap_avg_speed = sns.heatmap(reshaped_avg_speed, cmap='magma', annot=False, cbar=False, ax=ax[0])
    ax[0].set_title("Average Heatmap of Speed Data")
    ax[0].tick_params(labelbottom=False)
    ax[0].tick_params(bottom=False)

    # Plot the average z-score heatmap
    heatmap_avg_zscore = sns.heatmap(reshaped_avg_zscore, cmap='magma', annot=False, cbar=False, ax=ax[1])
    ax[1].set_title("Average Heatmap of Z-Score")
    ax[1].tick_params(labelbottom=False)
    ax[1].tick_params(bottom=False)

    # Plot the averaged z-score with 95% CI on ax[2]
    time_axis = np.arange(len(avg_zscore))  # Create a time axis
    ax[2].plot(time_axis, avg_zscore, color='blue', label='Mean Z-Score')
    ax[2].fill_between(time_axis, avg_zscore - ci95_zscore, avg_zscore + ci95_zscore, color='blue', alpha=0.2, label='95% CI')
    ax[2].set_title("Averaged Z-Score with 95% CI")
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("Z-Score")
    ax[2].legend(loc="upper right")
    ax[2].set_xlim([0, len(avg_zscore) - 1])
    ax[2].set_ylim([avg_zscore.min() - ci95_zscore.max(), avg_zscore.max() + ci95_zscore.max()])
    ax[2].margins(x=0, y=0)
    # Optional: Remove the bottom and left ticks if needed (for a cleaner plot)
    ax[2].tick_params(left=False, bottom=False)
    # Remove the frame (top and right spines)
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[2].spines['bottom'].set_visible(False)
    ax[2].tick_params(labelbottom=False)
    ax[2].tick_params(bottom=False)
    #plot_wavelet(ax[3], avg_sst_ripple, frequency_ripple, avg_power_ripple, Fs, colorBar=True, logbase=True)
    plot_wavelet(ax[3], avg_sst, frequency, avg_power, Fs, colorBar=True, logbase=True)
    ax[3].set_title("Average Power (Theta Band)")
    #set color bar
    cbar_ax = fig.add_axes([ax[0].get_position().x0, ax[3].get_position().y0 - 0.15, 
                            ax[0].get_position().width * 0.2, 0.01])
    plt.colorbar(heatmap_avg_speed.collections[0], cax=cbar_ax, orientation='horizontal')

    cbar_ax = fig.add_axes([ax[1].get_position().x0 + 0.2, ax[3].get_position().y0 - 0.15, 
                            ax[1].get_position().width * 0.2, 0.01])
    plt.colorbar(heatmap_avg_zscore.collections[0], cax=cbar_ax, orientation='horizontal')
    plt.tight_layout()
    plt.show()
    return -1
#%%
dpath='E:/workingfolder/Group D/1819287/speed_files_2sec/'
Fs=840
plot_heatmap_all(dpath,Fs)
#%%
# #Check theta band with pyPhotometry data
# folder ='F:/CheeseboardYY/GCaMP8m/1804115/Day5_photometry_CB/'
# # File name
# file_name = 'py_py_4115-2024-08-04-131222_0.csv'
# Fs=130
# '''Read csv file and calculate zscore of the fluorescent signal'''
# raw_signal,raw_reference,Cam_Sync=fp.read_photometry_data (folder, file_name, readCamSync=True,plot=True)
# '''Get zdFF directly'''
# #zscore_raw = fp.get_zdFF(raw_reference,raw_signal,smooth_win=2,remove=0,lambd=5e4,porder=1,itermax=50)
# zscore_smooth=fp.smooth_signal(raw_signal,window_len=10,window='flat')
# zscore_bandpass = band_pass_filter(raw_signal, 4, 60, Fs)
# #reshaped_zscore_bandpass=zscore_bandpass.reshape(1, -1)
# sst, frequency, power, _ = Calculate_wavelet(zscore_bandpass, lowpassCutoff=60, Fs=Fs, scale=2)
# fig, ax = plt.subplots(1, 1, figsize=(8, 4))
# plot_wavelet(ax, sst, frequency, power, Fs, colorBar=False, logbase=True)
# ax.set_title("Theta band")

# #%%
# #Check theta band with non-animal data
# file_path='F:/CheeseboardYY/Group D/Zscore_traceAll_1uW_SNR.csv'
# Fs=840
# df = pd.read_csv(file_path, header=None)
# zscore_raw = df[0].values[0:2*840]
# zscore_raw=notchfilter (zscore_raw,f0=100,bw=10,fs=Fs)
# zscore_smooth=fp.smooth_signal(zscore_raw,window_len=10,window='flat')
# zscore_bandpass = band_pass_filter(zscore_smooth, 4, 60, Fs)
# #reshaped_zscore_bandpass=zscore_bandpass.reshape(1, -1)
# sst, frequency, power, _ = Calculate_wavelet(zscore_bandpass, lowpassCutoff=100, Fs=Fs, scale=10)
# fig, ax = plt.subplots(1, 1, figsize=(8, 2))
# plot_wavelet(ax, sst, frequency, power, Fs, colorBar=False, logbase=True)
# ax.set_title("Theta band")

# time_axis=np.arange(len(zscore_smooth)) /Fs
# fig, ax = plt.subplots(1, 1, figsize=(8, 2))
# ax.plot(time_axis, zscore_smooth, color='k', label='Smoothed Z-Score')