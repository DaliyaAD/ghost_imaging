#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:35:14 2026

Calculates key pattern statistics as an entire array and stores them in a
callable dictionary. 

@author: ellaward
"""

import numpy as np
import scipy.fft
from skimage.measure import shannon_entropy as ent

eps = 1e-12


def fwhm(profile, centre):
    """
    Determines the FWHM by moving across profiles in steps until the values
    exceed a tolerance of 0.5.

    Parameters
    ----------
    profile : Array. 
    centre : Integer. Central element in the array.

    Returns
    -------
    FWHM

    """
    right = centre
    while right < len(profile) - 1 and profile[right] >= 0.5:
        right += 1

    left = centre
    while left > 0 and profile[left] >= 0.5:
        left -= 1
    return right - left


def autocorrelation(stack):
    """
    Calculates the autocorrelation width using Wiener Khinchin theorum

    Parameters
    ----------
    stack : 3D array. 

    Returns
    -------
    stack_fwhm : Float. Autocorrelation with for the 3D stack of patterns.

    """
    slice_fwhms = []
    M = len(stack)
    for i in range(M):
        slice_2d = stack[i]
        mean = np.mean(slice_2d)
        subt_mean = slice_2d - mean
        F = scipy.fft.fft2(subt_mean)
        power_spec = F * np.conj(F)
        # The real part of the 2D inverse Fourier transform gives autocorrelation
        autocorr = scipy.fft.ifft2(power_spec).real
        # Peak value is shifted to the centre
        autocorr = scipy.fft.fftshift(autocorr)
        centre_y, centre_x = np.array(autocorr.shape) // 2
        autocorr = autocorr / autocorr[centre_y, centre_x]
        # PROFILES
        h_profile = autocorr[centre_y, :]
        v_profile = autocorr[:, centre_x]
        # FWHM
        h_fwhm = fwhm(h_profile, centre_x)
        v_fwhm = fwhm(v_profile, centre_y)
        avg_fwhm = (h_fwhm + v_fwhm) // 2
        slice_fwhms.append(avg_fwhm)
    stack_fwhm = np.mean(slice_fwhms)
    return stack_fwhm


def mutual_coherence(A_matrix):
    """
    Calculates the mutual coherence between 

    Parameters
    ----------
    A_matrix : TYPE
        DESCRIPTION.

    Returns
    -------
    mutual_coherence : TYPE
        DESCRIPTION.

    """
    norms = np.linalg.norm(A_matrix, axis=0, keepdims=True)
    normalization = A_matrix / (norms+eps)
    gram_matrix = normalization.T @ normalization
    np.fill_diagonal(
        gram_matrix, 0)  # Don't assign to a variable as this function directly modifies gram_matrix.
    mutual_coherence = np.max(np.abs(gram_matrix))
    return mutual_coherence


def calc_pattern_contrast(array):
    """
    Function calculates contrast for a 2D image.

    Parameters
    ----------
    array : 3D array. Stack of patterns
    Returns
    -------
    pattern_contrast : Float. Contrast for one slide of a 3D array.

    """
    pattern_contrast = []
    for i in range(len(array)):
        slice_2d = array[i]
        contrast = np.std(slice_2d)/(np.mean(slice_2d) + eps)
        pattern_contrast.append(contrast)
    return pattern_contrast


def calc_pattern_entropy(array):
    """
    Function calculates entropy for a 2D image.

    Parameters
    ----------
    array : 3D array. Stack of patterns

    Returns
    -------
    pattern_entropy : Float. Entropy for one slice of a 3D array.

    """
    pattern_entropy = []
    for i in range(len(array)):
        slice_2d = array[i]
        entropy = ent(slice_2d)
        pattern_entropy.append(entropy)
    return pattern_entropy


def stack_statistics(stack):
    """
    Compiles relevant basic pattern statistics into a dictionary. Statistics computed
    are for the whole stack and not individual slices.

    Parameters
    ----------
    stack : 3D array. Stack of patterns.

    Returns
    -------
    dict
        Dictionary of key pattern statistics stored as floats.

    """
    return {
        "mean": float(np.mean(stack)),
        "var": float(np.var(stack)),
        "std": float(np.std(stack)),
        "contrast": float(np.std(stack)/(np.mean(stack) + eps)),
        "entropy": float(np.mean(calc_pattern_entropy(stack))),
        "autocorrelation width": float(autocorrelation(stack)),
    }


def effective_rank(A):
    s = np.linalg.svd(A, compute_uv=False)

    # Normalize singular values into probabilities
    p = s / np.sum(s)

    # Remove zeros to avoid log(0)
    p = p[p > 0]

    # Shannon entropy
    entropy = -np.sum(p * np.log(p))

    return np.exp(entropy)


def compute_matrix_stats(A):
    """
    Computes pattern statistics that rely upon the sensing matrix, A.

    Parameters
    ----------
    A : 2D array. Sensing matrix of the pattern stack.

    Returns
    -------
    dict
        Dictionary of key pattern statistics.

    """
    rank = effective_rank(A)
    singular_values = np.linalg.svd(A, compute_uv=False)
    cond_number = singular_values[0] / \
        singular_values[-1] if singular_values[-1] > 0 else np.inf

    return {
        "effective_rank": int(rank),
        "cond_number": float(cond_number),
        "mutual coherence": float(mutual_coherence(A)),
    }


def compute_all_stats(stack, A):
    """
    Updates an empty list with the newly computed pattern statistics.

    Parameters
    ----------
    stack : 3D array. 
    A : 2D array. 

    Returns
    -------
    stats : Dictionary. Complete list of key pattern statistics.

    """
    stats = {}
    stats.update(stack_statistics(stack))
    stats.update(compute_matrix_stats(A))
    return stats
