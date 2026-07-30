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
    'save_plot': True,
    'plot_file_name': 'SR_vs_SSIM_Ridge',
    'plot_title': f'The Effect of Sampling Ratio Against SSIM for {CONFIG["recon_type"]}',
    'x_label': 'Sampling Ratio (%)',
    'y_label': 'SSIM',
}


def create_plot(x_data, y_data):
    """
    Produces a scatterplot between two variables.

    Parameters
    ----------
    x_data : numpy array of floats
    y_data : numpy array of floats

    Returns
    -------
    None.

    """
    plot_file_name = SAVING_INFO['plot_file_name']
    plot_title = SAVING_INFO['plot_title']
    x_label = SAVING_INFO['x_label']
    y_label = SAVING_INFO['y_label']

    figure = plt.figure(figsize=(8, 9))
    plt.rcParams['font.family'] = 'Times New Roman'
    axes_main_plot = figure.add_subplot(211)
    axes_main_plot.grid()
    axes_main_plot.xaxis.set_major_locator(
        plt.MultipleLocator(25))
    axes_main_plot.scatter(x_data, y_data)
    axes_main_plot.set_title(plot_title, fontsize=14)
    axes_main_plot.set_xlabel(x_label, fontsize=18)
    axes_main_plot.set_ylabel(y_label, fontsize=18)
    axes_main_plot.tick_params(axis='both', labelsize=12)
    plt.savefig(plot_file_name, dpi=400)


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

x_data = df['Sampling Ratio (%)']
y_data = df['SSIM']
create_plot(x_data, y_data)
