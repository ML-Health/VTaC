"""
Filters the ABP, ECG, and PPG signals in the VT_ALARMS dataset.

ECG:
    - Highpass filter at 1 Hz
    - Lowpass filter at 30 Hz
    - Notch filter at 60 Hz
PPG:
    - Notch filter at 60 Hz
    - Bandpass filter at 0.5-5 Hz
ABP:
    - Notch filter at 60 Hz
    - Lowpass filter at 16 Hz
"""
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import torch
import sys

split = sys.argv[1]

samples, ys, names = torch.load(f"data/out/lead_selected/{split}.pt")
num_channels = samples.shape[1]

SAMPLING_FREQ = 250
POWERLINE_FREQ = 60


def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def notch_filter(freq, Q, fs):
    b, a = iirnotch(freq, Q, fs)
    return b, a


def filter_ecg_channel(data):
    b, a = butter_highpass(1.0, SAMPLING_FREQ)
    b2, a2 = butter_lowpass(30.0, SAMPLING_FREQ)
    tempfilt = filtfilt(b, a, data)
    tempfilt = filtfilt(b2, a2, tempfilt)
    b_notch, a_notch = notch_filter(POWERLINE_FREQ, 30, SAMPLING_FREQ)
    tempfilt = filtfilt(b_notch, a_notch, tempfilt)
    return tempfilt


def filter_ppg_channel(data):
    b_notch, a_notch = notch_filter(POWERLINE_FREQ, 30, SAMPLING_FREQ)
    tempfilt = filtfilt(b_notch, a_notch, data)
    N_bp, Wn_bp = butter(1, [0.5, 5], btype="band", analog=False, fs=SAMPLING_FREQ)
    tempfilt = filtfilt(N_bp, Wn_bp, tempfilt)
    return tempfilt


def filter_abp_channel(data):
    b_notch, a_notch = notch_filter(POWERLINE_FREQ, 30, SAMPLING_FREQ)
    tempfilt = filtfilt(b_notch, a_notch, data)
    b2, a2 = butter_lowpass(16.0, SAMPLING_FREQ)
    tempfilt = filtfilt(b2, a2, tempfilt)
    return tempfilt


output_samples = []

# Free up memory when we're done with the data
ecg_data_1 = np.array([filter_ecg_channel(x[0, :]) for x in samples])
output_samples.append(torch.from_numpy(ecg_data_1.copy()).float())
del ecg_data_1

ecg_data_2 = np.array([filter_ecg_channel(x[1, :]) for x in samples])
output_samples.append(torch.from_numpy(ecg_data_2.copy()).float())
del ecg_data_2

pleth_data = np.array([filter_ppg_channel(x[2, :]) for x in samples])
output_samples.append(torch.from_numpy(pleth_data.copy()).float())
del pleth_data

abp_data = np.array([filter_abp_channel(x[3, :]) for x in samples.numpy()])
output_samples.append(torch.from_numpy(abp_data.copy()).float())
del abp_data

output_samples = torch.stack(output_samples, dim=1)

output_samples = output_samples.float()

torch.save(
    (output_samples, ys, names),
    f"data/out/{split}-filtered.pt",
)
