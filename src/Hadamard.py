#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 10:55:17 2026

@author: ellaward
"""
from patterns_mat_A import pattern_setup
from patterns_mat_A import scale_normalize
from scipy.linalg import hadamard
import numpy as np
import matplotlib.pyplot as plt


def gen_hadamard_random_rows(CONFIG):
    """
    Creates a stack of Hadamard matricies by randomly chosing M rows (with no 
    repeats) and reshapes the array into a useable stack of patterns. 

    Currently uses min-max normalisation to shift from -1/1 to 0/1. 

    Parameters
    ----------
    CONFIG : Dictionary. Contains input data for several variables as
    decided by the user.

    Returns
    -------
    hard_pattern : Array. Stack of generated Hadamard matricies
    """
    samp_rat, arr_size, arr_seed, M, rng, stack_shape, A_shape = pattern_setup(
        CONFIG)
    N = arr_size**2
    H = hadamard(N)
    rows = rng.choice(N, size=M, replace=False)
    selected = H[rows]
    hard_pattern = selected.reshape(M, arr_size, arr_size)
    normalized_hard = scale_normalize(hard_pattern)
    return normalized_hard


def gen_hadamard_sequency(CONFIG):
    """
    Sequency can be though of as the number of sign changes as you move a long 
    a row of H_shape. (-1, 1, -1, 1, -1, 1, has a HIGH sequency whereas 
                       1, 1, 1, 1, -1, -1, has a LOW sequency)
    We will then choose the rows with highest sequency and lowest sequency to 
    vary the appearance of the Hadamard matrix. 

    'sequency' gives the number of sign changes in a row. 
    'ordered_sequency' gives the row numbers ordered by sequency so that the 
    extremities can be chosen. 

    Parameters
    ----------
    CONFIG : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    samp_rat, arr_size, arr_seed, M, rng, stack_shape, A_shape = pattern_setup(
        CONFIG)
    level = "high"
    N = arr_size**2
    H = hadamard(N)
    # Makes element-wise sign comparisons and sums over them.
    sequency = np.sum(H[:, :-1] != H[:, 1:], axis=1)
    ordered_sequency = np.argsort(sequency)

    if level == "low":
        rows = ordered_sequency[:M]
    elif level == "high":
        rows = ordered_sequency[-M:]
    else:
        raise ValueError(f"Unknown mode: {level}")
    selected = H[rows]
    hard_pattern = selected.reshape(M, arr_size, arr_size)
    normalized_hard = scale_normalize(hard_pattern)
    return normalized_hard


CONFIG = {
    'arr_size': 64,
    'seed': 48,
    'samp_rat': 80,
}

patt = gen_hadamard_sequency(CONFIG)


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

    plt.show()


for i in range(10):
    disp_pattern(patt[i])
