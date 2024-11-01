# Models

This folder contains code for defining, training, and evaluating supervised and contrastive learning models used in the [VTaC paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/7a53bf4e02022aad32a4019d41b3b476-Paper-Datasets_and_Benchmarks.pdf).  


The Convolutional Neural Network (CNN), Fully-Connected Neural Network (FCN), and their respective contrastive learning (CL) versions, including CNN+CL, and FCN+CL architectures and some of the code are adapted from the following work. 

* Zhou, Y., Zhao, G., Li, J., Sun, G., Qian, X., Moody, B., Mark, R. G., and Lehman, L. H. [A Contrastive Learning Approach for ICU False Arrhythmia Alarm Reduction.](https://rdcu.be/cJf9V) Nature Scientific Reports, 2022.

The SAE (Supervised Autoencoder) architecture and code is adapted from this work:

* Eric P. Lehman, Rahul G. Krishnan, Xiaopeng Zhao, Roger G. Mark, and Li-wei H. Lehman, [Representation Learning Approaches to Detect False Arrhythmia Alarms from ECG Dynamics.](https://proceedings.mlr.press/v85/lehman18a.html) Proceedings of the 3rd Machine Learning for Healthcare Conference, PMLR 85:571-586, 2018.

The Transformer implementation is based on the following work:

* Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. Attention Is All You Need. Advances in neural information processing
systems, 30, 2017

For generative approaches, including diffusion models, please see links provided in the VTaC paper.  Code for the rule-based approach can be downloaded from the [PhysioNet Challenge 2015 website](https://www.physionet.org/static/published-projects/challenge-2015/1.0.0/sources/).

# File Structure 

Each model folder has 4 files: 

## nets.py
This file contains the definitions of PyTorch neural networks used for the respective model. 

## tools.py

This is a utility file that provides functions for training and evaluation. It defines a custom PyTorch Dataset for training and maintains the signal data and target labels. This file is also adapted from Zhou et al 2022.

## eval.py

This is an evaluation script that loads models and runs evals on the test set. It either uses a fixed threshold or tunes the threshold on the score-maximizing value from the validation set before using it on the test set. 

## train.py

This is a training file that takes in hyperparameters based on the type of model. This code is adapted from Zhou et. al. 2022.


