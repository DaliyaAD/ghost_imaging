#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:31:40 2026

@author: ellaward
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
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


def sampling_to_M(samp_rat, arr_size):
    """Converts a sampling ratio (%) to number of patterns M."""
    return int((samp_rat * arr_size**2) / 100)


def pattern_setup(CONFIG):
    samp_rat = CONFIG['samp_rat']
    arr_size = CONFIG['arr_size']
    arr_seed = CONFIG['seed']
    M = sampling_to_M(samp_rat, arr_size)
    rng = np.random.default_rng(seed=arr_seed)
    stack, A = np.zeros((M, arr_size, arr_size)
                        ), np.zeros((M, arr_size**2))
    stack_shape, A_shape = stack.shape, A.shape
    return samp_rat, arr_size, arr_seed, M, rng, stack_shape, A_shape


def gen_binary(CONFIG):
    samp_rat, arr_size, arr_seed, M, rng, stack_shape, A_shape = pattern_setup(
        CONFIG)
    level = CONFIG['parameter_value']
    p_bright, p_dark = level/100, 1-(level/100)
    bin_pattern = rng.choice([0, 1], size=(
        M, arr_size, arr_size), p=[p_dark, p_bright])

    return bin_pattern


def gen_gaussian(CONFIG):
    samp_rat, arr_size, arr_seed, M, rng, stack_shape, A_shape = pattern_setup(
        CONFIG)
    level = CONFIG['parameter_value']
    if level == "low":
        contrast = 3
    elif level == "medium":
        contrast = 2
    elif level == "high":
        contrast = 1
    else:
        raise ValueError(
            f"Unknown contrast value '{level}'. Choose from 'low', 'medium', or 'high.")

    rng = np.random.default_rng(seed=arr_seed)
    arr = scale_normalize(rng.normal(
        loc=0.5, scale=0.5, size=(M, arr_size, arr_size)))
    z_scores = (arr - arr.mean()) / arr.std()
    z_clipped = np.clip(z_scores, -contrast, contrast)
    gaus_pattern = scale_normalize(z_clipped)

    return gaus_pattern


"""
Z scores measure how many standard deviations from a mean a value is.
Removing the outlier values and then redistributing the important signal data when normalising creates a higher contrast.
More data removed means more contrast when renormalising
"""


def gen_noise(CONFIG):
    samp_rat, arr_size, arr_seed, M, rng, stack_shape, A_shape = pattern_setup(
        CONFIG)
    level = CONFIG['parameter_value']

    if level == "small":
        blur = 0
    elif level == "moderate":
        blur = 1
    elif level == "large":
        blur = 2
    else:
        raise ValueError(
            f"Unknown grain size '{level}'. Choose from 'small', 'moderate', or 'large.")

    rng = np.random.default_rng(seed=arr_seed)
    arr = scale_normalize(rng.random(size=(M, arr_size, arr_size)))
    corr_pattern = gaussian_filter(arr, blur, axes=(1, 2))
    return corr_pattern


def disp_pattern(CONFIG, pattern):
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
    samp_rat = CONFIG['samp_rat']
    save_pattern_fig = CONFIG['save_pattern_fig']
    plt.figure(figsize=(5, 5))
    plt.imshow(pattern, cmap='gray')
    plt.title(f"Sparsity={samp_rat}%")
    plt.axis('off')
    if save_pattern_fig:
        plt.savefig(f"Phantom_recon_{samp_rat}.png", dpi=400)
    plt.show()


def gen_hadamard(CONFIG):
    samp_rat, arr_size, arr_seed, M, rng, stack_shape, A_shape = pattern_setup(
        CONFIG)
    base = hadamard(arr_size)  # keep the original matrix fixed
    hard_pattern = []
    for i in range(M):
        rng = np.random.default_rng(seed=i)
        variant = rng.permutation(base)
        hard_pattern.append(variant)
    return scale_normalize(hard_pattern)


def gen_matrix_A(stack):
    M, arr_size, arr_size = stack.shape
    mat_A = stack.reshape(M, arr_size**2)
    return mat_A


def build_metadata(CONFIG, pattern_type, stack):
    samp_rat = CONFIG['samp_rat']
    arr_size = CONFIG['arr_size']
    arr_seed = CONFIG['seed']
    param_value = CONFIG['parameter_value']
    M = stack.shape[0]

    matrix_id = f"{pattern_type.lower()}_{param_value:.2f}_size{arr_size}_M{M}_seed{arr_seed}"

    return {
        "matrix_id": matrix_id,
        "pattern_type": pattern_type,
        "parameter_value": param_value,
        "samp_rat": samp_rat,
        "arr_size": arr_size,
        "seed": arr_seed,
        "M": M,
        "N_pixels": arr_size**2,
    }


_GENERATORS = {
    "Binary": (gen_binary, "Binary mask density", lambda CONFIG: f"{CONFIG['binary_density']} %"),
    "Gaussian": (gen_gaussian, "Contrast", lambda CONFIG: CONFIG['gaussian_contrast']),
    "Correlated Noise": (gen_noise, "Grain size", lambda CONFIG: CONFIG['grain_size']),
}


def produce_pattern(CONFIG):
    pattern_type = CONFIG['pattern_type']
    try:
        gen_fn, extra_label, extra_value_fn = _GENERATORS[pattern_type]
    except KeyError:
        raise ValueError(
            f"Unknown pattern type '{pattern_type}'. Choose from {list(_GENERATORS)}.")

    stack = gen_fn(CONFIG)
    A = gen_matrix_A(stack)
    metadata = build_metadata(CONFIG, pattern_type.capitalize(), stack)

    return stack, A, metadata
