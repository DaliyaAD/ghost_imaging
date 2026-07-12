#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:43:50 2026

Simulates bucket detection using random binary patterns and customizable 
phantom image through basic correlative reconstruction.

Determines the fidelity of reconstruction through a few image metrics
(NMSE, PSNR, SSIM). These functions are taken from a pre-existing library 
(skimage.mertics) and are excecuted in metrics.py as of 8/6/26.


@author: Ella Ward
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from phantoms import make_phantom
from metrics import image_metric
import time
from tabulate import tabulate

# PLOTTING INFORMATION
PLOT_TITLE = 'The Effect of M Value on Image Reconstruction Quality'
X_LABEL = 'M value'
Y_LABEL = 'Mean Squared Error'

GRID_LINES = True
SAVE_FIGURE = True
FIGURE_NAME = 'Effect_of_M.png'
FIGURE_RESOLUTION = 400


def create_plot(x_data, y_data):
    """
    Produces a scatterplot between two variables.

    Parameters
    ----------
    x_data : numpy array of floats
    y_data : numpy array of floats

    Returns
    -------
    None.

    """
    figure = plt.figure(figsize=(8, 9))
    plt.rcParams['font.family'] = 'Times New Roman'
    axes_main_plot = figure.add_subplot(211)
    axes_main_plot.grid(GRID_LINES)
    axes_main_plot.xaxis.set_major_locator(
        plt.MultipleLocator(np.max(x_data) * 2))
    axes_main_plot.scatter(x_data, y_data)
    axes_main_plot.set_title(PLOT_TITLE, fontsize=14)
    axes_main_plot.set_xlabel(X_LABEL, fontsize=18)
    axes_main_plot.set_ylabel(Y_LABEL, fontsize=18)
    axes_main_plot.tick_params(axis='both', labelsize=12)
    plt.savefig(FIGURE_NAME, dpi=FIGURE_RESOLUTION)


def generate_pattern(pattern, M, arr_size=None, arr_seed=None):
    rng = np.random.default_rng(seed=arr_seed)
    if pattern == "binary":
        return rng.integers(0, 2, size=(M, arr_size, arr_size))
    elif pattern == "gaussian":
        return rng.normal(loc=0.5, scale=0.5, size=(M, arr_size, arr_size))
    else:
        raise ValueError(f"Unknown pattern '{pattern}'. "
                         f"Choose from 'binary', 'gaussian'.")


def compute_bucket_values(stack, phantom):
    """
    Calculates the product between a random speckled pattern and the phantom.

    Parameters
    ----------
    stack : ND array, stack of random binary patterns.
    phantom : 2D array, original binary image.

    Returns
    -------
    array, bucket value for each product array.

    """
    return np.sum(stack * phantom, axis=(1, 2))


def reconstruct_image(bucket_values, stack):
    """
    Reconstructs the image by taking element-wise covariance between the 
    bucket values and phantom

    Parameters
    ----------
    bucket_values : float, bucket value for each product array
    stack : ND array, stack of random binary patterns

    Returns
    -------
    ND array, reconstructed image

    """
    weighted = stack * bucket_values[:, None, None]
    mean_weighted = np.mean(weighted, axis=0)
    mean_bucket = np.mean(bucket_values)
    mean_stack = np.mean(stack, axis=0)
    return mean_weighted - mean_bucket * mean_stack


def scale_normalize(reconstructed_pattern):
    """
    Rescales the reconstructed image using min-max normalisation.

    Parameters
    ----------
    reconstructed_pattern : ND array, unscaled reconstructed image

    Returns
    -------
    normalised_pattern : ND array, scaled reconstructed image to match the phantom.

    """
    normalised_pattern = (reconstructed_pattern-reconstructed_pattern.min()) / \
        (reconstructed_pattern.max()-reconstructed_pattern.min())
    return normalised_pattern


def disp_pattern(pattern, M):
    """
    Displays an array as a greyscale image.

    Parameters
    ----------
    pattern : ND array, image being displayed
    M : float, number of speckled patterns (only used for plot title)

    Returns
    -------
    None.

    """
    plt.imshow(pattern, cmap='gray')
    plt.title(f"M={M}")
    plt.axis('off')
    plt.savefig(f"Phantom_recon_{M}.png", dpi=400)
    plt.show()


def compute_recons(phantom, pattern, arr_size, M, arr_seed=None):
    """
    Computes the image reconstructions over a series of M value intervals up 
    to a maximum. The M values and reconstructed arrays are assigned to a an
    array so that image metrics can be calculated and plots can be made
    between them.

    Parameters
    ----------
    phantom : ND array, phantom image
    arr_size : float, size of array
    interval : float, M interval 
    upper_m : float, upper M value
    arr_seed : float, seed of random speckled patterns

    Returns
    -------
    TYPE
        DESCRIPTION.
    recon : TYPE
        DESCRIPTION.

    """
    full_stack = generate_pattern(
        pattern, M, arr_size=arr_size, arr_seed=arr_seed)
    bucket_all = compute_bucket_values(full_stack, phantom)
    recon = scale_normalize(reconstruct_image(bucket_all, full_stack))
    #disp_pattern(recon, M)
    return recon


def main(CONFIG):

    # Parameters
    arr_size = CONFIG['arr_size']
    #pattern = CONFIG['pattern']
    phantom = make_phantom(CONFIG['phantom'], arr_size)
    samp_rat = CONFIG['samp_rat']
    seed = CONFIG['seed']

    results = []
    patterns = ['binary', 'gaussian']
    for pattern in patterns:
        for ratio in range(1, samp_rat, 1):
            for sd in range(1, seed):
                start = time.perf_counter()
                M = int(ratio * arr_size**2)         # use the loop variable
                recons = compute_recons(
                    phantom, pattern, arr_size, M, arr_seed=sd)   # use the loop variable
                rows = image_metric(
                    recons, phantom, CONFIG['phantom'], pattern, M, arr_size, sd, data_range=1)
                runtime = time.perf_counter() - start
                rows = rows + (runtime,)
                results.append(rows)

    headers = ["Phantom", "Pattern", "Image Size", "Seed",
               "Sampling ratio", "NMSE", "PSNR", "SSIM", "Runtime"]

    df = pd.DataFrame(results, columns=headers)
    data = df.to_csv("Exp_results.csv", index=False)
    table = (tabulate(df, headers=headers, tablefmt="grid"))
    print(table)
    return rows


CONFIG = {
    'arr_size': 32,
    'phantom': 'A',
    'samp_rat': 5,
    'seed': 3

}
main(CONFIG)
