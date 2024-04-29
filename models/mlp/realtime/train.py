import torch
import os
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from nets import *
import time
from tools import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
import sklearn
import sys

if __name__ == "__main__":
    SEED = 1 if len(sys.argv) <= 5 else int(sys.argv[5])
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.chdir("data/out/sample-norm")
    # load preprocessed dataset
    trainset_x, trainset_y, train_names = torch.load("train.pt")
    valset_x, valset_y, val_names = torch.load("val.pt")
    testset_x, testset_y, test_names = torch.load("test.pt")
    num_channels = trainset_x.shape[1]

    zero_nans = lambda x: torch.nan_to_num(x, 0)

    trainset_x = zero_nans(trainset_x)
    testset_x = zero_nans(testset_x)
    valset_x = zero_nans(valset_x)

    batch_size = int(sys.argv[1])
    lr = float(sys.argv[2])
    # dl = float(sys.argv[3])
    dropout_probability = float(sys.argv[3])
    positive_class_weight = float(sys.argv[4])

    params_training = {
        "framework": "mlp",
        "weighted_class": positive_class_weight,
        "learning_rate": lr,
        "adam_weight_decay": 0.005,
        "batch_size": batch_size,
        "max_epoch": 500,
        "data_length": 2500,
    }

    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    # save path of trained model

    out_name = f"{batch_size}-{lr}-{dropout_probability}-{positive_class_weight}-{SEED}"
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", out_name
    )

    if not any(
        os.path.exists(os.path.join(model_path, x)) for x in ["", "auc", "score"]
    ):
        os.makedirs(os.path.join(model_path, "auc"))
        os.makedirs(os.path.join(model_path, "score"))
    save_path = os.path.join(model_path, "results.txt")
    logger = get_logger(logpath=save_path, filepath=os.path.abspath(__file__))
    logger.info(params_training)

    model_save_path = os.path.join(
        model_path, str(params_training["learning_rate"]) + ".pt"
    )

    dataset_train = Dataset_train(trainset_x, trainset_y)
    dataset_eval = Dataset_train(valset_x, valset_y)
    dataset_test = Dataset_train(testset_x, testset_y)

    params = {
        "batch_size": params_training["batch_size"],
        "shuffle": False,
        "num_workers": 0,
    }

    iterator_train = DataLoader(dataset_train, **params)
    iterator_test = DataLoader(dataset_eval, **params)
    iterator_heldout = DataLoader(dataset_test, **params)

    model = ClassifierMLP(10000, dropout_prob=dropout_probability)

    logger.info(model)
    logger.info(
        "Num of Parameters: {}M".format(
            sum(x.numel() for x in model.parameters()) / 1000000
        )
    )

    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params_training["learning_rate"],
        weight_decay=params_training["adam_weight_decay"],
    )  # optimize all cnn parameters
    loss_ce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([params_training["weighted_class"]]).to(device)
    )

    num_epochs = params_training["max_epoch"]

    results_trainloss = []
    results_evalloss = []
    results_score = []
    results_TPR = []
    results_TNR = []
    results_acc = []
    max_score, max_auc = 0, 0
    min_eval_loss = float("inf")

    for t in range(1, 1 + num_epochs):
        train_loss = 0
        differ_loss_val = 0
        model = model.train()
        train_TP, train_FP, train_TN, train_FN = 0, 0, 0, 0

        for b, batch in enumerate(
            iterator_train, start=1
        ):  # signal_train, alarm_train, y_train, signal_test, alarm_test, y_test = batch
            loss, Y_train_prediction, y_train = train_model(
                batch, model, loss_ce, device, weight=0
            )
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= b
        eval_loss = 0
        model = model.eval()
        types_TP = 0
        types_FP = 0
        types_TN = 0
        types_FN = 0
        with torch.no_grad():
            for b, batch in enumerate(iterator_test, start=1):
                loss, Y_eval_prediction, y_test = eval_model(
                    batch, model, loss_ce, device
                )
                types_TP, types_FP, types_TN, types_FN = evaluation_test(
                    Y_eval_prediction, y_test, types_TP, types_FP, types_TN, types_FN
                )
                eval_loss += loss.item()

        eval_loss /= b
        acc = 100 * (types_TP + types_TN) / (types_TP + types_TN + types_FP + types_FN)
        score = (
            100
            * (types_TP + types_TN)
            / (types_TP + types_TN + types_FP + 5 * types_FN)
        )
        TPR = 100 * types_TP / (types_TP + types_FN)
        TNR = 100 * types_TN / (types_TN + types_FP)

        if types_TP + types_FP == 0:
            ppv = 1
        else:
            ppv = types_TP / (types_TP + types_FP)
        auc = sklearn.metrics.roc_auc_score(
            y_test.cpu().detach().numpy(), Y_eval_prediction.cpu().detach().numpy()
        )
        f1 = types_TP / (types_TP + 0.5 * (types_FP + types_FN))
        sen = types_TP / (types_TP + types_FN)
        spec = types_TN / (types_TN + types_FP)

        if score > max_score:
            max_score = score
            torch.save(
                model.state_dict(), os.path.join(model_path, "score", str(t) + ".pt")
            )

        if auc > max_auc:
            max_auc = auc
            torch.save(
                model.state_dict(), os.path.join(model_path, "auc", str(t) + ".pt")
            )

        logger.info(20 * "-")

        logger.info(params_training["framework"] + " Epoch " + str(t))

        logger.info(
            "total_loss: "
            + str(round(train_loss, 5))
            + " train_loss: "
            + str(round(train_loss, 5))
            + " eval_loss: "
            + str(round(eval_loss, 5))
        )

        logger.info(
            "TPR: "
            + str(round(TPR, 3))
            + " TNR: "
            + str(round(TNR, 3))
            + " Score: "
            + str(round(score, 3))
            + " Acc: "
            + str(round(acc, 3))
        )

        logger.info(
            "PPV: "
            + str(round(ppv, 3))
            + " AUC: "
            + str(round(auc, 3))
            + " F1: "
            + str(round(f1, 3))
        )
