import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def get_logger(
    logpath, filepath, package_files=[], displaying=True, saving=True, debug=False
):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="w")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


class Dataset_train(Dataset):
    # 'Characterizes a dataset for PyTorch'
    def __init__(self, signal_train, y_train):
        # signal
        self.strain = signal_train
        # groundtruth
        self.ytrain = y_train

    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.ytrain)

    def __getitem__(self, index):
        # 'Generates one sample of data'
        # Select sample
        return self.strain[index], self.ytrain[index]


def train_model(batch, model, loss_ce, device, weight):
    signal_train, y_train = batch
    signal_train = signal_train[:, :, 72500:75000].to(device)
    y_train = y_train.to(device)
    Y_train_prediction, reconstructed_signal = model(signal_train)
    loss = loss_ce(Y_train_prediction, y_train)
    criterion = nn.MSELoss()
    reconstruction_loss = criterion(signal_train, reconstructed_signal) ** 0.5
    return loss, weight * reconstruction_loss, Y_train_prediction, y_train


def eval_model(batch, model, loss_ce, device):
    signal_train, y_train = batch
    signal_train = signal_train[:, :, 72500:75000].to(device)

    y_train = y_train.to(device)

    # prediction
    Y_train_prediction, _ = model(signal_train)

    loss = loss_ce(Y_train_prediction, y_train)

    return loss, Y_train_prediction, y_train


def evaluation(Y_eval_prediction, y_test, TP, FP, TN, FN):
    # set 0 if false alarm else 1
    pre = (Y_eval_prediction >= 0).int()
    # pair: (prediction, groundtruth)
    for i, j in zip(pre, y_test):
        if i.item() == 1 and j.item() == 1:
            TP += 1
        if i.item() == 1 and j.item() == 0:
            FP += 1
        if i.item() == 0 and j.item() == 0:
            TN += 1
        if i.item() == 0 and j.item() == 1:
            FN += 1
    return TP, FP, TN, FN


def evaluate_rule_based(rule_based_results, y_test):
    TP = FP = TN = FN = 0
    # pair: (prediction, groundtruth)
    for i, j in zip(rule_based_results, y_test):
        if i.item() == 1 and j.item() == 1:
            TP += 1
        if i.item() == 1 and j.item() == 0:
            FP += 1
        if i.item() == 0 and j.item() == 0:
            TN += 1
        if i.item() == 0 and j.item() == 1:
            FN += 1
    return (
        100 * TP / (TP + FN),
        100 * TN / (TN + FP),
        100 * (TP + TN) / (TP + TN + FP + 5 * FN),
        100 * (TP + TN) / (TP + TN + FP + FN),
    )


def evaluation_test(Y_eval_prediction, y_test, types_TP, types_FP, types_TN, types_FN):
    pre = (Y_eval_prediction >= 0).int()
    for i, j in zip(pre, y_test):
        if i.item() == 1 and j.item() == 1:
            types_TP += 1
        if i.item() == 1 and j.item() == 0:
            types_FP += 1
        if i.item() == 0 and j.item() == 0:
            types_TN += 1
        if i.item() == 0 and j.item() == 1:
            types_FN += 1
    return types_TP, types_FP, types_TN, types_FN


def evaluate_raise_threshold(
    prediction, groundtruth, types_TP, types_FP, types_TN, types_FN, threshold
):
    prediction = torch.sigmoid(prediction)
    pre = 1 if prediction >= threshold else 0
    # identify if a sample is TP, FP, TN, or FN
    if pre == 1 and groundtruth == 1:
        types_TP += 1
    elif pre == 1 and groundtruth == 0:
        types_FP += 1
    elif pre == 0 and groundtruth == 1:
        types_FN += 1
    elif pre == 0 and groundtruth == 0:
        types_TN += 1

    return types_TP, types_FP, types_TN, types_FN


def eval(batch, model, device):
    signal_train, y_train = batch
    length = 2500
    signal_train = signal_train[:, :, 75000 - length : 75000].to(device)
    y_train = y_train.to(device)
    Y_train_prediction = model(signal_train)
    return Y_train_prediction, y_train
