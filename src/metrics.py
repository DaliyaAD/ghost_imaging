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

from CGI_sim import compute_recons as cr
from phantoms import make_phantom as mp
from CGI_sim import image_metric as im
from CGI_sim import normal_MSE as NMSE
from tabulate import tabulate
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from CGI_sim import disp_pattern

ARR_SIZE = 32
SHAPE, SHAPE_PARAMS = "sine", {
    "amplitude": 15, "frequency": 0.2, "thickness": 2}
INTERVAL, UPPER_M = 1000, 5000
SEED = 43
# Make phantom and generate reconstructed images
phantom = mp(SHAPE, ARR_SIZE, **SHAPE_PARAMS)
disp_pattern(phantom, 3000)
M_values, recons = cr(phantom, ARR_SIZE, interval=INTERVAL,
                      upper_m=UPPER_M, arr_seed=SEED)

# Image metrics
nmse = im(NMSE, recons, phantom, data_range=1)
psnr = im(PSNR, recons, phantom, data_range=1)
ssim = im(SSIM, recons, phantom, data_range=1)
sam_rat = M_values/(ARR_SIZE**2)
# Make table
headers = ["M value", "Sampling ratio", "NMSE", "PSNR", "SSIM"]
rows = list(zip(M_values, sam_rat, nmse, psnr, ssim))

print(tabulate(rows, headers=headers, tablefmt="grid"))
