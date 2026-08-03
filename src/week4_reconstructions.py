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
from phantoms import make_phantom

headers = ["Phantom", "Pattern", "Sampling Ratio (%)", "NMSE",
           "PSNR", "SSIM", "Seed", "Image Size", "Runtime"]
rows = []

CONFIG = {
    'arr_size': 32,
    'seed': 48,
    'samp_rat': 80,
    'phantom_shape': 'Shepp-Logan',
    'pattern_type': 'Binary',
    'parameter_value': 50,
    'recon_type': ["CGI", "DGI", "Ridge"]
}

SAVING_INFO = {
    'save_data': True,
    'data_file_name': 'Comparison_recon_algorithms',
    'save_pattern_image': True,
    'image_title': f'{CONFIG["recon_type"]}',
    'image_file_name': f'Example_recon_{CONFIG["recon_type"]}',
    'save_plot': False,
    'plot_file_name': 'SR_vs_SSIM_DGI',
    'plot_title': f'The Effect of Sampling Ratio Against SSIM for {CONFIG["recon_type"]}',
    'x_label': 'Sampling Ratio (%)',
    'y_label': 'SSIM',
}


def disp_pattern(pattern, image_title, file_name):
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
    plt.title(image_title)
    if SAVING_INFO['save_pattern_image']:
        plt.savefig(file_name, dpi=400)
    plt.show()


for reconstruction in CONFIG['recon_type']:
    results, recon = main(CONFIG, reconstruction)
    image_title = reconstruction
    file_name = f"W4_example_{reconstruction}.png"
    disp_pattern(recon, image_title, file_name)
    rows.append(results)


headers = ["Reconstruction Type", "Phantom", "Pattern", "Sampling Ratio (%)", "NMSE",
           "PSNR", "SSIM", "Seed", "Image Size", "Runtime"]
df = pd.DataFrame(rows, columns=headers)
if SAVING_INFO['save_data']:
    data = df.to_csv(f"{SAVING_INFO['data_file_name']}.csv", index=False)
table = (tabulate(df, headers=headers, tablefmt="grid"))

phantom_shape = CONFIG['phantom_shape']
arr_size = CONFIG["arr_size"]
phantom = make_phantom(phantom_shape, arr_size)
disp_pattern(phantom, "Example Shepp-Logan", 'Example_Shepp_Logan.png')
