import torch
from nets import FCN
from tools import evaluate_raise_threshold
from sklearn.metrics import roc_auc_score
import sys
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")


def max_file_in_directories(path):
    """
    Finds the file with the maximum numeric value in each directory within the given path.
    :param path: The path of the folder to inspect.
    :return: A dictionary where keys are directory names and values are the files with the maximum numeric value.
    """
    max_files = {}

    # Get all directories in the given path
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for directory in directories:
        full_directory_path = os.path.join(path, directory, "score")
        files = [f for f in os.listdir(full_directory_path) if os.path.isfile(os.path.join(full_directory_path, f))]

        if files:  # If there are files in the directory
            # Filter out 'results.txt' and convert filenames to integers for comparison
            filtered_files = [x for x in files if x != "results.txt"]
            if filtered_files:
                max_file = max(filtered_files, key=lambda f: int(f.rsplit('.', 1)[0]))
                max_files[directory] = os.path.join(full_directory_path, max_file)
            else:
                max_files[directory] = None
        else:
            max_files[directory] = None  # No files in the directory

    return max_files

run_to_model_path = max_file_in_directories(model_path)
print(run_to_model_path)
X_val, y_val, name_val = torch.load("data/out/val.pt")
X_val = X_val[:, :, 72500:75000].to(torch.device("cuda"))
X_test, y_test, name_test = torch.load("data/out/test.pt")
X_test = X_test[:, :, 72500:75000].to(torch.device("cuda"))


def find_best_threshold_parallel(y_pred, y_test):
    possible_thresholds = np.linspace(0, 1, 30000)
    best_score = -np.inf
    best_thresh = None

    # Ensure y_pred and y_test are numpy arrays
    y_pred = np.asarray(y_pred)
    y_test = np.asarray(y_test)

    # Initialize arrays to store TP, FP, TN, FN for each threshold
    TP = np.zeros_like(possible_thresholds)
    FP = np.zeros_like(possible_thresholds)
    TN = np.zeros_like(possible_thresholds)
    FN = np.zeros_like(possible_thresholds)

    # Vectorize the calculation over all thresholds
    for i, thresh in enumerate(possible_thresholds):
        # Get predictions for this threshold
        preds = y_pred >= thresh
        # Calculate TP, FP, TN, FN
        TP[i] = np.sum((preds == 1) & (y_test == 1))
        FP[i] = np.sum((preds == 1) & (y_test == 0))
        TN[i] = np.sum((preds == 0) & (y_test == 0))
        FN[i] = np.sum((preds == 0) & (y_test == 1))

    # Calculate score for each threshold
    scores = 100 * (TP + TN) / (TP + TN + FP + 5 * FN)

    # Find best score and threshold
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_thresh = possible_thresholds[best_idx]

    return best_thresh, best_score

accs = []
scores = []
TPRs = []
TNRs = []
ppvs = []
f1s = []
aucs = []
thresholds = []
val_scores = []
runs = []

for run, path in tqdm(run_to_model_path.items()):
    model = FCN(128, 256, 128, 51, 25, 13, 4)
    model.load_state_dict(torch.load(path, map_location=torch.device("cuda")))
    model = model.eval().to("cuda")

    y_pred_val = model(X_val)
    y_pred_test = model(X_test)

    thresh, val_score = find_best_threshold_parallel(
        torch.sigmoid(y_pred_val).cpu().detach(), y_val.cpu().detach().numpy()
    )
    print(thresh, val_score)
    types_TP, types_FP, types_TN, types_FN = (0, 0, 0, 0)
    for pred, gt in zip(y_pred_test, y_test):
        types_TP, types_FP, types_TN, types_FN = evaluate_raise_threshold(
            pred, gt, types_TP, types_FP, types_TN, types_FN, thresh
        )

    acc = 100 * (types_TP + types_TN) / (types_TP + types_TN + types_FP + types_FN)
    score = (
        100 * (types_TP + types_TN) / (types_TP + types_TN + types_FP + 5 * types_FN)
    )
    TPR = types_TP / (types_TP + types_FN)
    TNR = types_TN / (types_TN + types_FP)

    if types_TP + types_FP == 0:
        ppv = 1
    else:
        ppv = types_TP / (types_TP + types_FP)
    cpu_device = torch.device("cpu")
    auc = roc_auc_score(y_test.cpu().numpy(), y_pred_test.cpu().detach().numpy())
    f1 = types_TP / (types_TP + 0.5 * (types_FP + types_FN))
    runs.append(run)
    accs.append(acc)
    scores.append(score)
    TPRs.append(TPR)
    TNRs.append(TNR)
    ppvs.append(ppv)
    f1s.append(f1)
    aucs.append(auc)
    thresholds.append(thresh)
    val_scores.append(val_score)

    del model
    del y_pred_val
    del y_pred_test
    torch.cuda.empty_cache()

data = {
    "hyperparams": runs,
    "tpr": TPRs,
    "tnr": TNRs,
    "ppv": ppvs,
    "f1": f1s,
    "score": scores,
    "val_score": val_scores,
    "auc": aucs,
    "accuracy": accs,
    "threshold": thresholds,
}


df = pd.DataFrame(data)
df.to_csv(out_path, index=False)
