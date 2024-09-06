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
        position = fig.add_axes([0.5, -0.05, 0.4, 0.05])
        #position = fig.add_axes()
        cbar=plt.colorbar(CS, cax=position, orientation='horizontal', fraction=0.05, pad=0.5)
        cbar.set_label('Power (mV$^2$)', fontsize=12) 
        #plt.subplots_adjust(right=0.7, top=0.9)              
    return -1

#%%
dpath='G:/CheeseboardYY/Group D/1819287/speed_files/Day5/'
filename='Speed_lpfw_Day5-3.csv'
Fs=840
filepath=os.path.join(dpath,filename)
df = pd.read_csv(filepath)
sample_number=len(df)
midpoint=sample_number//2

speed_data = df['instant_speed'].values
speed_data[speed_data > 20] = np.nan
speed_series = pd.Series(speed_data)
speed_series.interpolate(method='nearest', inplace=True)
speed_data=speed_series.to_numpy()
reshaped_speed = speed_data.reshape(1, -1)
# Prepare data for wavelet analysis
zscore_raw = -df['raw_z_score'].values
zscore_raw=notchfilter (zscore_raw,f0=100,bw=10,fs=840)
zscore_smooth=fp.smooth_signal(zscore_raw,window_len=10,window='flat')
reshaped_zscore = zscore_smooth.reshape(1, -1)
zscore_bandpass = band_pass_filter(zscore_smooth, 4, 100, Fs)
# Calculate the wavelet transform
sst, frequency, power, global_ws = Calculate_wavelet(zscore_bandpass, lowpassCutoff=100, Fs=Fs, scale=10)
reshaped_zscore_bandpass=zscore_bandpass.reshape(1, -1)
#%%
fig, ax = plt.subplots(3, 1, figsize=(12, 4),gridspec_kw={'height_ratios': [1, 1, 3]})
heatmap =sns.heatmap(reshaped_speed, cmap='magma', annot=False, cbar=False, ax=ax[0])
ax[0].set_title("Heatmap of Speed Data")
ax[0].tick_params(labelbottom=False)  # Remove x-tick labels
ax[0].tick_params(bottom=False)  # Remove x-ticks


heatmap_zscore =sns.heatmap(reshaped_zscore, cmap='magma', annot=False, cbar=False, ax=ax[1])
ax[1].set_title("Heatmap of Zscore")
ax[1].tick_params(labelbottom=False)  # Remove x-tick labels
ax[1].tick_params(bottom=False)  # Remove x-ticks

# Plot wavelet analysis on the second axis (ax[1])
plot_wavelet(ax[2], sst, frequency, power, Fs, colorBar=True, logbase=True)
ax[2].set_title("Theta band")
cbar_ax = fig.add_axes([ax[0].get_position().x0, ax[2].get_position().y0 - 0.15, 
                        ax[0].get_position().width*0.2, 0.03])  # Adjust the position of the colorbar
plt.colorbar(heatmap.collections[0], cax=cbar_ax, orientation='horizontal')

cbar_ax = fig.add_axes([ax[1].get_position().x0+0.2, ax[2].get_position().y0 - 0.15, 
                        ax[1].get_position().width*0.2, 0.03])  # Adjust the position of the colorbar
plt.colorbar(heatmap_zscore.collections[0], cax=cbar_ax, orientation='horizontal')
plt.show()
#%%
fig, ax = plt.subplots(figsize=(12, 3))
zscore_bandpass=band_pass_filter(zscore_smooth,130,180,Fs)
sst,frequency,power,global_ws=Calculate_wavelet(zscore_bandpass,lowpassCutoff=200,Fs=Fs,scale=4)
plot_wavelet(ax,sst,frequency,power,Fs,colorBar=False,logbase=False)
#%%
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(zscore_smooth)
zscore_bandpass = butter_filter(zscore_raw, btype='low', cutoff=150, fs=Fs, order=5)
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(zscore_bandpass)

