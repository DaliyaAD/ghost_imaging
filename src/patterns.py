#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:31:40 2026

Contains function for binary, gaussian, correlated noise, and Hadamard pattern
stacks all normalized to contain values from 0 to 1.

Saves stack features in a dictionary with a matrix id. 

@author: ellaward
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.linalg import hadamard

statistics = []


def scale_normalize(pattern):
    """
    Rescales an array of values using min-max normalisation.

    Parameters
    ----------
    pattern : ND array, unscaled reconstructed image

    Returns
    -------
    normalised_pattern : ND array, scaled reconstructed image to match the phantom.

    """
    arr_max = pattern.max(axis=(1, 2), keepdims=True)
    arr_min = pattern.min(axis=(1, 2), keepdims=True)
    arr_range = arr_max - arr_min
    normalized_pattern = np.divide(
        pattern - arr_min,
        arr_range,
        out=np.zeros_like(pattern, dtype=float),
        where=(arr_range != 0)
    )
    return normalized_pattern


def sampling_to_M(samp_rat, arr_size):
    """
    Converts the desired sampling ratio into an M value which dictates the 
    length of the pattern stack.

    Parameters
    ----------
    samp_rat : Integer. Percentage of sample patterns against the size of the
    image.
    arr_size : Integer. Width and length of the array. Must be consuistent
    between phantom and pattern.

    Returns
    -------
    Integer. M value.

    """
    return int((samp_rat * arr_size**2) / 100)


def pattern_setup(CONFIG):
    """
    The pattern families are set up using similar/ the same code. 
    This function provides a general foundation for the stack generation by 
    determining and returning key characteristics.

    Parameters
    ----------
    CONFIG : Dictionary. Contains input data for several variables as
    decided by the user.

    Returns
    -------
    samp_rat : integer. Percentage of sample patterns against the size of the 
    image
    arr_size : Integer. Width and length of the array. Must be consistent 
    between phantom and pattern.
    arr_seed : Integer. Initializes the random generator. 
    M : Integer. Length of the pattern array.
    rng : Instance of the generator. 
    stack_shape : Array. The dimensions of the pattern stack.
    A_shape : Array. The dimensions of the pattern stack as a sensing matrix.

    """
    samp_rat = CONFIG['samp_rat']
    arr_size = CONFIG['arr_size']
    arr_seed = CONFIG['seed']
    level = CONFIG['parameter_value']
    M = sampling_to_M(samp_rat, arr_size)
    rng = np.random.default_rng(seed=arr_seed)
    stack, A = np.zeros((M, arr_size, arr_size)
                        ), np.zeros((M, arr_size**2))
    stack_shape, A_shape = stack.shape, A.shape
    return samp_rat, arr_size, arr_seed, M, rng, stack_shape, A_shape, level


def gen_binary(CONFIG):
    """
    Generates a stack of binary patterns with adjustable mask density.

    Parameters
    ----------
    CONFIG : Dictionary. Contains input data for several variables as
    decided by the user.

    Returns
    -------
    bin_pattern : Array. Stack of random binary patterns with length M.

    """
    samp_rat, arr_size, arr_seed, M, rng, stack_shape, A_shape, level = pattern_setup(
        CONFIG)

    p_bright, p_dark = level/100, 1-(level/100)
    bin_pattern = rng.choice([0, 1], size=(
        M, arr_size, arr_size), p=[p_dark, p_bright])

    return bin_pattern


def gen_gaussian(CONFIG):
    """
    Generates a stack of gaussian patterns with adjustable contrast.

    Parameters
    ----------
    CONFIG : Dictionary. Contains input data for several variables as
    decided by the user.

    Returns
    -------
    gaus_pattern : Array. Stack of randomly generated gaussian patterns with 
    length M.

    """
    samp_rat, arr_size, arr_seed, M, rng, stack_shape, A_shape, level = pattern_setup(
        CONFIG)

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
    """
    Generates a stack of correlated noise patterns with adjustable grain size.

    Parameters
    ----------
    CONFIG : Dictionary. Contains input data for several variables as
    decided by the user.

    Returns
    -------
    corr_pattern : Array. Stack of randomly generated correlared noise patterns 
    with length M.

    """
    samp_rat, arr_size, arr_seed, M, rng, stack_shape, A_shape, level = pattern_setup(
        CONFIG)

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
    arr = rng.random(size=(M, arr_size, arr_size))
    corr_pattern = gaussian_filter(arr, blur, axes=(1, 2))
    return scale_normalize(corr_pattern)


def gen_hadamard(CONFIG):
    samp_rat, arr_size, arr_seed, M, rng, stack_shape, A_shape, level = pattern_setup(
        CONFIG)
    N = arr_size**2
    H = hadamard(N)
    sequency = np.sum(H[:, :-1] != H[:, 1:], axis=1)
    ordered_sequency = np.argsort(sequency)

    if level == "random":
        rows = rng.choice(N, size=M, replace=False)
        selected = H[rows]
    elif level == "low":
        rows = ordered_sequency[:M]
    elif level == "high":
        rows = ordered_sequency[-M:]
    else:
        raise ValueError(
            f"Unknown mode: {level}. Choose from 'random', 'low', 'high'")

    selected = H[rows]
    norm_selected = (selected+1) / 2
    hard_pattern = norm_selected.reshape(M, arr_size, arr_size)
    return hard_pattern


def gen_matrix_A(stack):
    """
    Converts the stack of patterns into a 2D array to be used as a sensing
    matrix.

    Parameters
    ----------
    stack : 3D array. Stack of patterns

    Returns
    -------
    mat_A : 2D array.

    """
    M, arr_size, arr_size = stack.shape
    mat_A = stack.reshape(M, arr_size**2)
    return mat_A


def build_metadata(CONFIG, pattern_type, stack):
    """
    Writes a dictionary of stack characteristics including a matrix id for
    file identification and experiment recreation.

    Parameters
    ----------
    CONFIG : Dictionary. Contains input data for several variables as
    decided by the user.
    pattern_type : String. Pattern family name.
    stack : 3D array. Stack of patterns

    Returns
    -------
    dict
        Features of the pattern stack.

    """
    samp_rat = CONFIG['samp_rat']
    arr_size = CONFIG['arr_size']
    arr_seed = CONFIG['seed']
    param_value = CONFIG['parameter_value']
    M = stack.shape[0]

    return{
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
    "Hadamard": (gen_hadamard, "Sequency", lambda CONFIG: CONFIG['sequency']),
}


def produce_pattern(CONFIG):
    """
    Parameters
    ----------
    CONFIG : Dictionary. Contains input data for several variables as
    decided by the user.

    Raises
    ------
    ValueError
        Fails if the inputted pattern type is not in the code. 

    Returns
    -------
    stack : 3D array. Stack of patterns
    A : Integer. Length of the sensing matrix.
    metadata : Dictionary. Metadata of the pattern stack.
    """
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
