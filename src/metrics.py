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


def image_metric(recons, phantom):
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
    return {
        "nmse": float(normal_MSE(recons, phantom, data_range=1)),
        "psnr": float(PSNR(recons, phantom, data_range=1)),
        "ssim": float(SSIM(recons, phantom, data_range=1)),
    }
