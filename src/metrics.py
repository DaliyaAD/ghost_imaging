#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:49:17 2026

Generates a phantom image based on desired parameters and makes a series of 
reconstructions based on chosen values for M.

Array size, seed number, phantom shape, and M values are all customizable.

Generates a table of values for different image metrics.

@author: Ella Ward
"""

import numpy as np
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from tabulate import tabulate


def normal_MSE(reconstruction, phantom, data_range=None):
    """
    Computes the mean squared error and normalises it by the mean value of the phantom

    Parameters
    ----------
    reconstruction : ND array, reconstructed but unnormalised image
    phantom : ND array, phantom image

    Returns
    -------
    NMSE : float, normalised mean squared error

    """
    NMSE = MSE(reconstruction, phantom) / np.mean(phantom**2)
    return NMSE


def image_metric(recons, phantom, phantom_type, pattern, M, arr_size, arr_seed, data_range=None):
    """
    Computes image metrics and assigns values to an array with the same indexing
    as M value.

    Parameters
    ----------
    metric : function, image metric
    recon : ND array, reconstructed image
    phantom : ND array, phantom image

    Returns
    -------
    values : ND array, metric values for different M values
    """
    nmse = normal_MSE(recons, phantom, data_range=1)
    psnr = PSNR(recons, phantom, data_range=1)
    ssim = SSIM(recons, phantom, data_range=1)
    sam_rat = M/(arr_size**2)
    rows = (phantom_type, pattern, arr_size**2,
            arr_seed, sam_rat, nmse, psnr, ssim)
    return rows
