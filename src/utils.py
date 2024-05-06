import torch
import numpy as np

# -----------------------------------------------------------------------------
# error_fn
# -----------------------------------------------------------------------------
def error_fn(df):
    pred = df.loc[:, "pred"]
    ground_truth = df.loc[:, "target"]
    weight = 1/30 + ground_truth

    return np.sum(((pred - ground_truth)**2) * weight, axis=0) / np.sum(weight, axis=0)


# -----------------------------------------------------------------------------
# metric_fn
# -----------------------------------------------------------------------------
def metric_fn(female, male):
    err_male = error_fn(male)
    err_female = error_fn(female)
    return (err_male + err_female) / 2 + abs(err_male - err_female)


# -----------------------------------------------------------------------------
# define_device
# -----------------------------------------------------------------------------
def define_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    return device