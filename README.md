# VTaC: A Benchmark Dataset of Ventricular Tachycardia Alarms from ICU Monitors

This repository contains code and script for the VTaC NeurIPS 2023 paper: 

Li-wei Lehman, Benjamin Moody, Harsh Deep, Feng Wu, Hasan Saeed, Lucas McCullum, Diane Perry, Tristan Struja, Qiao Li, Gari Clifford, Roger Mark, [VTaC: A Benchmark Dataset of Ventricular Tachycardia Alarms from ICU Monitors,](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7a53bf4e02022aad32a4019d41b3b476-Abstract-Datasets_and_Benchmarks.html) Advances in Neural Information Processing Systems 36 ([NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023)), Datasets and Benchmarks Track.
[PDF version.](https://proceedings.neurips.cc/paper_files/paper/2023/file/7a53bf4e02022aad32a4019d41b3b476-Paper-Datasets_and_Benchmarks.pdf)

##Abstract

False arrhythmia alarms in intensive care units (ICUs) are a continuing problem
despite considerable effort from industrial and academic algorithm developers. Of
all life-threatening arrhythmias, ventricular tachycardia (VT) stands out as the
most challenging arrhythmia to detect reliably. We introduce a new annotated
VT alarm database, VTaC (Ventricular Tachycardia annotated alarms from ICUs)
consisting of over 5,000 waveform recordings with VT alarms triggered by bedside
monitors in the ICUs. Each VT alarm in the dataset has been labeled by at least
two independent human expert annotators. The dataset encompasses data collected
from ICUs in three major US hospitals and includes data from three leading bedside
monitor manufacturers, providing a diverse and representative collection of alarm
waveform data. Each waveform recording comprises at least two electrocardiogram
(ECG) leads and one or more pulsatile waveforms, such as photoplethysmogram
(PPG or PLETH) and arterial blood pressure (ABP) waveforms. We demonstrate
the utility of this new benchmark dataset for the task of false arrhythmia alarm
reduction, and present performance of multiple machine learning approaches,
including conventional supervised machine learning, deep learning, contrastive
learning and generative approaches for the task of VT false alarm reduction. 

The VTaC dataset will be released on [PhysioNet](physionet.org) in Summer 2024.
Contact: [Li-wei Lehman](https://web.mit.edu/lilehman/www/), LILEHMAN <AT> MIT.EDU


## Setup

Python 3.8 dependencies:
 - torch 1.9
 - wfdb 4.1.1
 - scipy
 - matplotlib

Install dependencies:\
`pip install -r requirements.txt`

## Preprocessing

Apply filtering on the dataset:

`python preprocessing/filtering.py [split]`

Applies the following filters for each lead type:
- ECG:
    - 60 Hz notch
    - 1 Hz highpass
    - 30 Hz lowpass
- PPG
    - 60 Hz notch
    - 0.5-5 Hz bandpass

- ABP

Standardize the data by removing outliers and z-score normalizing:\
`python preprocessing/standardize.py`

 - Z-score normalizes on a per-event basis, using the event's mean and standard deviation for each channel.

## Model Traning and Evaluation
Experiments are located under `models/[model-name]/[realtime or retrospective]`
To train a model, use the `train.py` script with the hyperparameters.

For example to train an MLP on the realtime task use `python models/mlp/realtime/train.py 64 0.0001 0.1 4`. To evaluate model performance, run the `eval.py` script which will evaluate on all models in the trained models output directory and produce a csv with the results. 


