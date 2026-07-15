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

import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from phantoms import make_phantom
from metrics import image_metric
import time
from tabulate import tabulate
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
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


def generate_pattern(pattern, M, arr_size=None, arr_seed=None):
    rng = np.random.default_rng(seed=arr_seed)
    if pattern == "binary":
        return rng.integers(0, 2, size=(M, arr_size, arr_size))
    elif pattern == "gaussian":
        return scale_normalize(rng.normal(loc=0.5, scale=0.5, size=(M, arr_size, arr_size)))
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


def disp_pattern(pattern, samp_rat):
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
    plt.figure(figsize=(5, 5))
    plt.imshow(pattern, cmap='gray')
    plt.title(f"Sampling ratio={samp_rat}%")
    plt.axis('off')
    plt.savefig(f"Phantom_recon_{samp_rat}.png", dpi=400)
    plt.show()


def sampling_to_M(samp_rat, arr_size):
    """Converts a sampling ratio (%) to number of patterns M."""
    return int((samp_rat * arr_size**2) / 100)


def compute_recon(phantom, pattern, arr_size, samp_rat, arr_seed=None):
    """
    Generates a pattern stack at a single fixed sampling ratio and
    returns the reconstructed image.
    """
    M = sampling_to_M(samp_rat, arr_size)
    full_stack = generate_pattern(
        pattern, M, arr_size=arr_size, arr_seed=arr_seed)
    bucket_all = compute_bucket_values(full_stack, phantom)
    recon = reconstruct_image(bucket_all, full_stack)
    return recon, M


def main(CONFIG):
    arr_size = CONFIG['arr_size']
    phantom_type = CONFIG['phantom']
    phantom = make_phantom(phantom_type, arr_size)
    samp_rat = CONFIG['samp_rat']       # now a single fixed value, not a max
    seed_step = CONFIG.get('seed_step', 10)
    num_seeds = CONFIG['num_seeds']
    patterns = CONFIG.get('patterns', ['binary', 'gaussian'])

    seeds = range(1, num_seeds, seed_step)
    results = []
    table = []

    for pattern, sd in itertools.product(patterns, seeds):
        start = time.perf_counter()
        recon, M = compute_recon(
            phantom, pattern, arr_size, samp_rat, arr_seed=sd)
        recon = scale_normalize(recon)
        disp_pattern(recon, samp_rat)

        row = image_metric(
            recon, phantom, phantom_type, pattern, M, arr_size, sd, data_range=1)
        runtime = time.perf_counter() - start
        results.append(row + (runtime,))
        nmse, psnr, ssim = row[3], row[4], row[5]
        table.append([samp_rat, pattern, sd, nmse, psnr, ssim])

    headers = ["Phantom", "Pattern", "Sampling ratio (%)",
               "NMSE", "PSNR", "SSIM", "Seed", "Array size", "Runtime (s)"]
    tab_headers = ["Sampling ratio (%)", "Pattern",
                   "Seed", "NMSE", "PSNR", "SSIM"]
    df = pd.DataFrame(results, columns=headers)
    df.to_csv(f"Results_{phantom_type}_{samp_rat}%.csv", index=False)
    print(tabulate(table, headers=tab_headers, tablefmt="grid"))
    return df


CONFIG = {
    'arr_size': 32,
    'phantom': 'A',
    'samp_rat': 25,      # fixed sampling ratio (%), not a max/sweep anymore
    'num_seeds': 3,
    'seed_step': 1,
    'patterns': ['binary', 'gaussian'],
}
main(CONFIG)
