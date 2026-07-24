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

#from patterns import produce_pattern
import numpy as np
import matplotlib.pyplot as plt
from metrics import image_metric
import time
from patterns_mat_A import produce_pattern
from phantoms import make_phantom


# PLOTTING INFORMATION
PLOT_TITLE = 'The Effect of M Value on Image Reconstruction Quality'
X_LABEL = 'M value'
Y_LABEL = 'Mean Squared Error'

GRID_LINES = True


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


def compute_ref_values(stack):
    return np.sum(stack, axis=(1, 2))


def diff_signal(bucket_values, reference_values):
    scale_factor = np.mean(bucket_values) / np.mean(reference_values)
    return bucket_values - scale_factor * reference_values


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


def disp_pattern(pattern):
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
    plt.axis('off')
    plt.savefig("Phantom_recon.png", dpi=400)
    plt.show()


def disp_pattern_2(var_name, var_value, pattern_type, pattern):
    plt.figure(figsize=(5, 5))
    plt.imshow(pattern[0], cmap='gray')
    plt.title(f"{var_name}={var_value}")
    plt.savefig(f"{pattern_type}_{var_value}.png", dpi=400)
    plt.axis('off')
    plt.show()


def sampling_to_M(samp_rat, arr_size):
    """Converts a sampling ratio (%) to number of patterns M."""
    return int((samp_rat * arr_size**2) / 100)


def compute_recon_CGI(phantom, pattern_stack):
    bucket_all = compute_bucket_values(pattern_stack, phantom)
    recon_CGI = reconstruct_image(bucket_all, pattern_stack)
    return recon_CGI


def compute_recon_DGI(phantom, pattern_stack):
    reference_values = compute_ref_values(pattern_stack)
    bucket_values = compute_bucket_values(pattern_stack, phantom)
    differential_signal = diff_signal(bucket_values, reference_values)
    recon_DGI = reconstruct_image(differential_signal, pattern_stack)
    return recon_DGI


def main(CONFIG, recon_type):

    arr_size = CONFIG['arr_size']
    seed = CONFIG['seed']
    phantom_shape = CONFIG['phantom_shape']
    phantom = make_phantom(phantom_shape, arr_size)
    pattern_type = CONFIG['pattern_type']
    samp_rat = CONFIG['samp_rat']
    M = sampling_to_M(samp_rat, arr_size)

    pattern_stack, A, metadata = produce_pattern(CONFIG)

    if recon_type == "CGI":
        recon = scale_normalize(compute_recon_CGI(phantom, pattern_stack))
    elif recon_type == "DGI":
        recon = compute_recon_DGI(phantom, pattern_stack)
    start = time.perf_counter()
    row = image_metric(
        recon, phantom, phantom_shape, pattern_stack, pattern_type, M, arr_size, seed, data_range=1)
    runtime = time.perf_counter() - start

    results = (recon_type,) + row + (runtime,)
    return results, recon
