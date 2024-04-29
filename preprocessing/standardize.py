"""
Z-score normalization and per-sample clipping
"""
import torch
from rich import print
from rich.progress import track
from matplotlib import pyplot as plt
from typing import Tuple


def compute_mean_std():
    samples, _, _ = torch.load("data/out/train-filtered.pt")
    samples = samples[:, :, 72500:75000]
    mu = []
    sigma = []
    # for each signal
    for i in range(samples.shape[1]):
        vals = []
        for x in range(len(samples)):
            sample = samples[x, i]
            # if current signal is available
            if sample.sum() != 0.0:
                vals.append(sample)

        vals = torch.cat(vals)
        mu.append(vals.mean())
        sigma.append(vals.std())
    return mu, sigma


def create_splits(mu, sigma):
    """
    Use the population mean and standard deviation to normalize each sample
    Saves the output to out/population-norm/{split}.pt

    Args:
        mu: list of population means for each channel
        sigma: list of population standard deviations for each channel

    Returns:
        None
    """
    for split in ["train", "val", "test"]:
        samples, ys, names = torch.load(f"data/out/{split}-filtered.pt")
        num_channels = samples.shape[1]
        for i in range(num_channels):
            mu_i = mu[i]
            sigma_i = sigma[i]
            for x in track(
                range(len(samples)), description="Normalizing...", transient=True
            ):
                if samples[x, i].sum() != 0.0:
                    samples[x, i] = (samples[x, i] - mu_i) / sigma_i

        samples = samples.float()
        torch.save((samples, ys, names), f"data/out/population-norm/{split}.pt")


def create_individual_splits():
    """
    Normalize each sample individually
    Saves the output to data/out/sample-norm/{split}.pt
    """
    for split in ["train", "val", "test"]:
        samples, ys, names = torch.load(f"data/out/{split}-filtered.pt")
        num_channels = samples.shape[1]
        for i in range(num_channels):
            for x in track(
                range(len(samples)), description="Normalizing...", transient=True
            ):
                if samples[x, i].sum() != 0.0:
                    mu_i = samples[x, i, 72500:75000].mean()
                    sigma_i = samples[x, i, 72500:75000].std()
                    samples[x, i] = (samples[x, i] - mu_i) / sigma_i

        samples = samples.float()
        torch.save((samples, ys, names), f"data/out/sample-norm/{split}.pt")


if __name__ == "__main__":
    print("Use population normalization or per-sample normalization?")
    print("1. Population normalization")
    print("2. Per-sample normalization")
    choice = input("Enter your choice (1 or 2):")
    if choice == "1":
        mu, sigma = compute_mean_std()
        create_splits(mu, sigma)
    elif choice == "2":
        create_individual_splits()
    else:
        print("Invalid choice. Exiting...")
        exit(1)
