# Preprocessing scripts

## filtering.py
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

Assumes a sampling freq of 250Hz and powerline freq of 60Hz.

## standardize.py

Normalizes the data based data from 10 seconds prior to VT alarm (samples 72500 to 75000). 
Can either normalize using the mean and stdev of the train set or on the individual sample.
