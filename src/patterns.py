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

statistics = []


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


def gen_binary(CONFIG):
    M = CONFIG['M']
    arr_size = CONFIG['arr_size']
    arr_seed = CONFIG['seed']
    density = CONFIG['binary_density']
    rng = np.random.default_rng(seed=arr_seed)
    p_bright, p_dark = density/100, 1-(density/100)
    bin_pattern = scale_normalize(rng.choice([0, 1], size=(M,
                                                           arr_size, arr_size), p=[p_dark, p_bright]))
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
        contrast = 1
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
    arr_seed = None
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

    rng = np.random.rand(M, arr_size, arr_size)
    corr_pattern = gaussian_filter(rng, blur)
    statistics = ["Random noise", "Grain size", label, M, arr_size, arr_seed,
                  np.mean(corr_pattern), np.var(corr_pattern)]
    return statistics, corr_pattern


def produce_pattern(CONFIG):
    pattern = CONFIG['pattern_type']
    if pattern == "binary":
        return gen_binary(CONFIG)
    elif pattern == "gaussian":
        return gen_gaussian(CONFIG)
    elif pattern == "random_noise":
        return gen_noise(CONFIG)
    else:
        raise ValueError(
            f"Unknown grain size '{pattern}'. Choose from 'binary', 'gaussian', or 'random noise.")


def disp_pattern(var_name, var_value, pattern_type, pattern):
    plt.figure(figsize=(5, 5))
    plt.imshow(pattern[0], cmap='gray')
    plt.title(f"{var_name}={var_value}")
    plt.savefig(f"{pattern_type}_{var_value}.png", dpi=400)
    plt.axis('off')
    plt.show()


CONFIG = {
    'arr_size': 32,
    'seed': 43,
    'M': 1,
    'pattern_type': 'gaussian',
    'grain_size': 'large',
    'gaussian_contrast': 'high',
    'binary_density': 40
}

results = list(produce_pattern(CONFIG)[0])
row = [produce_pattern(CONFIG)[0]]

disp_pattern(results[1], results[2], results[0], produce_pattern(CONFIG)[1])
stats_names = ['pattern_type', 'parameter_name', 'parameter_value',
               'M', 'arr_size', 'seed', 'mean_intensity', 'variance']

table = tabulate(row, headers=stats_names, tablefmt="grid")
print(table)
