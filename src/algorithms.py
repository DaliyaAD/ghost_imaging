#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:43:50 2026


@author: Ella Ward
"""

#from patterns import produce_pattern
import numpy as np
from metrics import image_metric
import time
from patterns import produce_pattern
from phantoms import make_phantom


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
    """
    Computes the sum of each 2D pattern in a 3D stack.

    Parameters
    ----------
    stack : 3D array. Stack of patterns.

    Returns
    -------
    Array. Reference values for each pattern.

    """
    return np.sum(stack, axis=(1, 2))


def diff_signal(bucket_values, reference_values):
    """
    Caluclates the differential signal between the phantom and the patterns.

    Parameters
    ----------
    bucket_values : Array. Sums of array products between phantom and pattern. 
    reference_values : Array. Sums of patterns.

    Returns
    -------
    Array. Differential signal value for each pattern.

    """
    scale_factor = np.mean(bucket_values) / np.mean(reference_values)
    return bucket_values - scale_factor * reference_values


def reconstruct_image(signal_values, stack):
    """
    Reconstructs the image by taking element-wise covariance between the 
    bucket values and phantom

    Parameters
    ----------
    signal_values : array, signal value for each pattern with respect to the
    phantom
    stack : ND array, stack of patterns

    Returns
    -------
    ND array, reconstructed image

    """
    weighted = stack * signal_values[:, None, None]
    mean_weighted = np.mean(weighted, axis=0)
    mean_signal = np.mean(signal_values)
    mean_stack = np.mean(stack, axis=0)
    return mean_weighted - mean_signal * mean_stack


def sampling_to_M(samp_rat, arr_size):
    """
    Converts a percentage sampling ratio to the length, M of the pattern array.

    Parameters
    ----------
    samp_rat : Integer. Percentage of sampling versus image size
    arr_size : Integer. Width and height of the image.

    Returns
    -------
    Integer. M value (array length)

    """
    return int((samp_rat * arr_size**2) / 100)


def compute_recon_CGI(phantom, pattern_stack):
    """
    Computes computational ghost imaging reconstruction.

    Parameters
    ----------
    phantom : 2D array. Phantom image.
    pattern_stack : 3D array. Stack of patterns.

    Returns
    -------
    recon_CGI : 2D array. Reconstructed image of the phantom using CGI.

    """
    bucket_all = compute_bucket_values(pattern_stack, phantom)
    recon_CGI = scale_normalize(reconstruct_image(bucket_all, pattern_stack))
    return recon_CGI


def compute_recon_DGI(phantom, pattern_stack):
    """
    Computes differential ghost imaging reconstruction.

    Parameters
    ----------
    phantom : 2D array. Phantom image.
    pattern_stack : 3D array. Stack of patterns.

    Returns
    -------
    recon_DGI : 2D array. Reconstructed image of the phantom using DGI.

    """
    reference_values = compute_ref_values(pattern_stack)
    bucket_values = compute_bucket_values(pattern_stack, phantom)
    differential_signal = diff_signal(bucket_values, reference_values)
    recon_DGI = scale_normalize(reconstruct_image(
        differential_signal, pattern_stack))
    return recon_DGI


def compute_recon_ridge(phantom, pattern_stack, A, lam):
    bucket_values = compute_bucket_values(pattern_stack, phantom)
    N = A.shape[1]

    overlap = A.T @ A  # Calculates the overlap between the pixels by taking the transpose
    reconstruct = A.T @ bucket_values  # Correlates patterns with bucket values
    t_hat = np.linalg.solve(overlap + lam * np.eye(N), reconstruct)
    # This solves the linear equation
    # lam * np.eye(N) scales an identity matrix by lambda, as per the equation
    # See page 61 in Hansen
    recon_ridge = scale_normalize(t_hat.reshape(phantom.shape))
    # The solution will still be 2D so we have to convert it back
    return recon_ridge


def main(CONFIG, recon_type):
    """
    Produces the pattern stack and reconstructs the phantom using chosen 
    algorithm.
    Computes image metrics and stores the run time the reconstruction process.

    Parameters
    ----------
    CONFIG : Dictionary. Contains input data for several variables as
    decided by the user.
    recon_type : String. Name of the chosen reconstruction algorithm

    Returns
    -------
    results : List. Image metrics.
    recon : 2D array. Reconstructed image of the phantom.

    """
    arr_size = CONFIG['arr_size']
    phantom_shape = CONFIG['phantom_shape']
    phantom = make_phantom(phantom_shape, arr_size)

    pattern_stack, A, metadata = produce_pattern(CONFIG)

    if recon_type == "CGI":
        recon = compute_recon_CGI(phantom, pattern_stack)
    elif recon_type == "DGI":
        recon = compute_recon_DGI(phantom, pattern_stack)
    elif recon_type == "Ridge":
        lam = CONFIG.get('lam', 1.0)          # add 'lam' to your CONFIG dict
        recon = compute_recon_ridge(phantom, pattern_stack, A, lam)
    else:
        raise ValueError(
            f"Unknown reconstruction algorithm '{recon_type}'.")

    start = time.perf_counter()
    metrics = image_metric(recon, phantom)
    runtime = time.perf_counter() - start
    metrics = {**metrics, 'runtime': runtime, }
    return metrics, recon
