import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


# -----------------------------------------------------------------------------
# calculate_mean_std
# -----------------------------------------------------------------------------
def calculate_mean_std(df):
    """
    Calculate the mean and standard deviation of the dataset.

    Args:
        df (pd.DataFrame): Dataframe with the filenames.

    Returns:
        mean, std (tuple): Mean and standard deviation of the dataset.
    """
    transform = transforms.ToTensor()
    sum_channels = torch.zeros(3)
    sum_channels_squared = torch.zeros(3)
    n_pixels = 0

    for _, row in df.iterrows():
        filename = row['filename']
        img = Image.open(f"./data/crops_100K/{filename}")
        img_tensor = transform(img)

        sum_channels += img_tensor.sum(dim=(1, 2))
        sum_channels_squared += (img_tensor ** 2).sum(dim=(1, 2))
        n_pixels += img_tensor.numel() // 3  # Each image has 3 channels

    mean = sum_channels / n_pixels
    std = torch.sqrt(sum_channels_squared / n_pixels - mean ** 2)
    
    return mean, std


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


# -----------------------------------------------------------------------------
# print_error_decile
# -----------------------------------------------------------------------------
def print_error_decile(results_df):
    """
    Print the mean error by gender decile.

    Args:
    - results_df (pd.DataFrame): DataFrame with evaluation results containing 'filename', 'pred', 'target' and 'gender' columns
    """
    error_decile = []
    for i in range(10):
        condition = (i / 10 <= results_df["gender"]) & (results_df["gender"] <= (i + 1) / 10)
        filtered_df = results_df.loc[condition]
        error_decile.append(error_fn(filtered_df))
    
    # Print results
    print("Mean error by gender decile:")
    for i in range(10):
        print(f"- Gender between {i * 0.1:.1f} and {(i + 1) * 0.1:.1f} : {error_decile[i]:.7f}")


# -----------------------------------------------------------------------------
# plot_images_with_info
# -----------------------------------------------------------------------------
def plot_images_with_info(df, title, savename):
    """
    Plot images with information and save to a file.

    Args:
    - df (pd.DataFrame): DataFrame with the images and info to plot.
    - title (str): Title of the plot.
    - savename (str): Filename to save the plot.
    """
    fig, ax = plt.subplots(1, len(df), figsize=(len(df) * 5, 5))
    fig.suptitle(title, fontsize=16)
    
    if len(df) == 1:
        ax = [ax]  # Ensure ax is iterable if there's only one subplot

    for i, (index, row) in enumerate(df.iterrows()):
        filename = row['filename']
        img = Image.open(f"./data/crops_100K/{filename}")
        ax[i].imshow(img)
        ax[i].axis('off')
        ax[i].set_title(
            f"Filename: {row['filename']}\n"
            f"Prediction: {row['pred']:.3f}, Target: {row['target']:.3f}, Gender: {row['gender']:.3f}", 
            fontsize=10
        )    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(savename)
    plt.close()


# -----------------------------------------------------------------------------
# print_best_worst_error
# -----------------------------------------------------------------------------
def print_best_worst_error(results_df):
    """
    Show best 5 pictures and worst 5 pictures in terms of error.
    Below each picture, print filename, prediction, target, and gender.

    Args:
    - results_df (pd.DataFrame): DataFrame with evaluation results containing 'filename', 'pred', 'target', and 'gender' columns.
    """
    # Calculate the absolute error
    results_df['error'] = (results_df['pred'] - results_df['target'])**2

    # Sort the DataFrame by error
    sorted_df = results_df.sort_values(by='error')

    # Select the best 5 and worst 5
    best_5 = sorted_df.head(5)
    worst_5 = sorted_df.tail(5)

    # Plot and save the best 5 results
    plot_images_with_info(best_5, "Best 5 pictures (lowest error)", "best_5.png")

    # Plot and save the worst 5 results
    plot_images_with_info(worst_5, "Worst 5 pictures (highest error)", "worst_5.png")