# VTaC: A Benchmark Dataset of Ventricular Tachycardia Alarms from ICU Monitors

This repository contains code and script for the VTaC NeurIPS 2023 paper: 

[Li-wei Lehman, Benjamin Moody, Harsh Deep, Feng Wu, Hasan Saeed, Lucas McCullum, Diane Perry, Tristan Struja, Qiao Li, Gari Clifford, Roger Mark, VTaC: A Benchmark Dataset of Ventricular Tachycardia Alarms from ICU Monitors,  Advances in Neural Information Processing Systems 36 (NeurIPS 2023) Datasets and Benchmarks Track. ](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7a53bf4e02022aad32a4019d41b3b476-Abstract-Datasets_and_Benchmarks.html) 

##Abstract

False arrhythmia alarms in intensive care units (ICUs) are a continuing problem despite considerable effort from industrial and academic algorithm developers. Of all life-threatening arrhythmias, ventricular tachycardia (VT) stands out as the most challenging arrhythmia to detect reliably. We introduce a new annotated VT alarm database, VTaC (Ventricular Tachycardia annotated alarms from ICUs) consisting of over 5,000 waveform recordings with VT alarms triggered by bedside monitors in the ICU. Each VT alarm waveform in the dataset has been labeled by at least two independent human expert annotators. The dataset encompasses data collected from ICUs in two major US hospitals and includes data from three leading bedside monitor manufacturers, providing a diverse and representative collection of alarm waveform data. Each waveform recording comprises at least two electrocardiogram (ECG) leads and one or more pulsatile waveforms, such as photoplethysmogram (PPG or PLETH) and arterial blood pressure (ABP) waveforms. We demonstrate the utility of this new benchmark dataset for the task of false arrhythmia alarm reduction, and present performance of multiple machine learning approaches, including conventional supervised machine learning, deep learning, semi-supervised learning, and generative approaches for the task of VT false alarm reduction.


## Setup

Python 3.8 dependencies:
 - torch 1.9
 - wfdb 4.1.1
 - scipy
 - matplotlib

Install dependencies:\
`pip install -r requirements.txt`

## Preprocessing

Split dataset into unanimous(all annotators agree) or adjudiated(adjudicator provided final decision) events:\
`python preprocessing/split_by_decision_type.py`

Create train/test/val folders and add event decision to metadata:\
`python preprocessing/split_event_data.py`

Analyze each event and only extract data from leads that are not flat:\
`python preprocessing/process_leads.py [split]`

Extracts 2 ECG channels, an ABP channel, and a PPG channel if the patient has that lead type. Uses a lead priority list for each type to determine which leads to take in the case there are multiple. For example, in a patient with lead II, lead V, and avR, the script will take leads II and V according to the priority list. Moreover, if there is only one ECG lead, that lead will be duplicated to both ECG channels. If a patient does not have a lead for that channel, it will be NaN.

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
    - 60 Hz notch
    - 16 Hz lowpass

Standardize the data by removing outliers and z-score normalizing:\
`python preprocessing/standardize.py`

 - Z-score normalizes on a per-event basis, using the event's mean and standard deviation for each channel.

## Model Traning and Evaluation
Experiments are located under `models/[model-name]/[realtime or retrospective]`
To train a model, use the `train.py` script with the hyperparameters.

For example to train an MLP on the realtime task use `python models/mlp/realtime/train.py 64 0.0001 0.1 4`. To evaluate model performance, run the `eval.py` script which will evaluate on all models in the trained models output directory and produce a csv with the results. 


## Analysis
Plot signals from the given split:\
`python visualizations/plot_signals.py [split]`

Plot a specific event
`python visualizations/plot_signal.py [event]`

Compute stats for the tables in the paper:
- Table 1: `python visualizations/table_1.py`
- Tables 2 and 4: `python visualizations/table_2_4.py`
- Table 3: `python visualizations/table_3.py` 
