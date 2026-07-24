#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 13:29:11 2026

@author: ellaward
"""

from DGI_w4 import main
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from patterns import produce_pattern


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


headers = ["Phantom", "Pattern", "Sampling Ratio (%)", "NMSE",
           "PSNR", "SSIM", "Seed", "Image Size", "Runtime"]
rows = []

CONFIG = {
    'arr_size': 64,
    'seed': 48,
    'samp_rat': 200,
    'phantom_shape': 'Shepp-Logan',
    'pattern_type': 'Binary',
    'parameter_value': 80,
    'save_data': False,
    'save_pattern_fig': False,
    'recon_type': ["CGI", "DGI"]
}


for reconstruction in CONFIG['recon_type']:
    results, recon = main(CONFIG, reconstruction)
    disp_pattern(recon)
    rows.append(results)


headers = ["Reconstruction Type", "Phantom", "Pattern", "Sampling Ratio (%)", "NMSE",
           "PSNR", "SSIM", "Seed", "Image Size", "Runtime"]
df = pd.DataFrame(rows, columns=headers)
table = (tabulate(df, headers=headers, tablefmt="grid"))
print(table)
