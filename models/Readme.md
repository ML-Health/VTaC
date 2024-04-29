# Models

Code for defining, training, and evaluating models. Each model folder has 4 files:

## nets.py
This file contains the definitions of PyTorch neural networks used for the respective model. The CNN, FCN, CNN+CL, and FCN+CL architectures and some of the code are adapted from the following work:

Zhou, Y., Zhao, G., Li, J., Sun, G., Qian, X., Moody, B., Mark, R. G., & Lehman, L. H. (2022). A Contrastive Learning Approach for ICU False Arrhythmia Alarm Reduction. Nature Scientific Reports.

The SAE architecture and code is adapted from this work:

"Representation Learning Approaches to Detect False Arrhythmia Alarms from ECG Dynamics," Eric P. Lehman, Rahul G. Krishnan, Xiaopeng Zhao, Roger G. Mark, Li-wei H. Lehman, Proceedings of the 3rd Machine Learning for Healthcare Conference, PMLR 85:571-586, 2018.

## tools.py

This is a utility file that provides functions for training and evaluation. It defines a custom PyTorch Dataset for training and maintains the signal data and target labels. This file is also adapted from Zhou et al 2022.

## eval.py

This is an evaluation script that loads models and runs evals on the test set. It either uses a fixed threshold or tunes the threshold on the score-maximizing value from the validation set before using it on the test set. 

## train.py

This is a training file that takes in hyperparameters based on the type of model. This code is adapted from Zhou et. al. 2022.
