#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:31:40 2026

@author: ellaward
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tabulate import tabulate
from scipy.linalg import hadamard

statistics = []


def scale_normalize(pattern):
    """
    Rescales the reconstructed image using min-max normalisation.

    Parameters
    ----------
    pattern : ND array, unscaled reconstructed image

    Returns
    -------
    normalised_pattern : ND array, scaled reconstructed image to match the phantom.

    """
    arr_max = pattern.max(axis=(1, 2), keepdims=True)
    arr_min = pattern.min(axis=(1, 2), keepdims=True)
    range = arr_max - arr_min
    normalised_pattern = (pattern-arr_min) / \
        (range)
    return normalised_pattern


def gen_binary(CONFIG):
    M = CONFIG['M']
    arr_size = CONFIG['arr_size']
    arr_seed = CONFIG['seed']
    density = CONFIG['binary_density']

    rng = np.random.default_rng(seed=arr_seed)
    p_bright, p_dark = density/100, 1-(density/100)
    bin_pattern = rng.choice([0, 1], size=(
        M, arr_size, arr_size), p=[p_dark, p_bright])

    statistics = ["Binary", "Binary mask density", f'{density} %', M, arr_size,
                  arr_seed, np.mean(bin_pattern), np.var(bin_pattern)]
    return statistics, bin_pattern


def gen_gaussian(CONFIG):
    M = CONFIG['M']
    arr_size = CONFIG['arr_size']
    arr_seed = CONFIG['seed']
    label = CONFIG['gaussian_contrast']

    if label == "low":
        contrast = 3
    elif label == "medium":
        contrast = 2
    elif label == "high":
        contrast = 1
    else:
        raise ValueError(
            f"Unknown contrast value '{label}'. Choose from 'low', 'medium', or 'high.")

    rng = np.random.default_rng(seed=arr_seed)
    arr = scale_normalize(rng.normal(
        loc=0.5, scale=0.5, size=(M, arr_size, arr_size)))
    z_scores = (arr - arr.mean()) / arr.std()
    z_clipped = np.clip(z_scores, -contrast, contrast)
    gaus_pattern = scale_normalize(z_clipped)
    statistics = ["Gaussian", "Contrast", label, M, arr_size, arr_seed,
                  np.mean(gaus_pattern), np.var(gaus_pattern)]
    return statistics, gaus_pattern


"""
Z scores measure how many standard deviations from a mean a value is.
Removing the outlier values and then redistributing the important signal data when normalising creates a higher contrast.
More data removed means more contrast when renormalising
"""


def gen_noise(CONFIG):
    M = CONFIG['M']
    arr_size = CONFIG['arr_size']
    arr_seed = CONFIG['seed']
    label = CONFIG['grain_size']

    if label == "small":
        blur = 0
    elif label == "moderate":
        blur = 1
    elif label == "large":
        blur = 2
    else:
        raise ValueError(
            f"Unknown grain size '{label}'. Choose from 'small', 'moderate', or 'large.")

    rng = np.random.default_rng(seed=arr_seed)
    arr = scale_normalize(rng.random(size=(M, arr_size, arr_size)))
    corr_pattern = gaussian_filter(arr, blur, axes=(1, 2))
    statistics = ["Correlated noise", "Grain size", label, M, arr_size, arr_seed,
                  np.mean(corr_pattern), np.var(corr_pattern)]
    return statistics, corr_pattern


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
    plt.title(f"Sparsity={samp_rat}%")
    plt.axis('off')
    plt.savefig(f"Phantom_recon_{samp_rat}.png", dpi=400)
    plt.show()


def gen_hadamard(CONFIG):
    arr_size = CONFIG['arr_size']
    M = CONFIG['M']
    base = hadamard(arr_size)  # keep the original matrix fixed
    arr = []
    for i in range(M):
        rng = np.random.default_rng(seed=i)
        # fresh permutation of the base matrix each time
        variant = rng.permutation(base)
        arr.append(variant)
        # pass the actual array, not the None from .append()
        #disp_pattern(variant, 1)
    statistics = ["Hadamard matrix", "None", None, M, arr_size, None,
                  np.mean(arr), np.var(arr)]
    return statistics, arr


def produce_pattern(CONFIG):
    pattern = CONFIG['pattern_type']
    if pattern == "binary":
        return gen_binary(CONFIG)
    elif pattern == "gaussian":
        return gen_gaussian(CONFIG)
    elif pattern == "correlated noise":
        return gen_noise(CONFIG)
    elif pattern == "hadamard":
        return gen_hadamard(CONFIG)
    else:
        raise ValueError(
            f"Unknown grain size '{pattern}'. Choose from 'binary', 'gaussian', 'correlated noise, or 'hadamard'.")
