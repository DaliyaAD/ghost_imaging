#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 09:55:27 2026

@author: ellaward
"""


from algorithms import main
import pandas as pd
import matplotlib.pyplot as plt
import matrix_stats
import patterns
from itertools import product

rows = []

CONFIG = {
    'arr_size': 32,
    'seed': [47, 48, 49],
    'samp_rat': [10, 20, 30, 40, 50, 60, 70, 80, 90],
    'phantom_shape': 'Shepp-Logan',
    'pattern_type': 'Binary',
    'parameter_value': 50,
    'recon_type': ["CGI", "DGI", "Ridge"]
}

PARAM_VALUES = {
    "Binary": [50],
    "Gaussian": ["medium"],
    "Correlated Noise": ["moderate"],
    "Hadamard": ["random", "low", "high"],
}

SAVING_INFO = {
    'save_data': False,
    'data_file_name': 'large_dataset',
    'save_pattern_image': False,
    'image_title': f'{CONFIG["recon_type"]}',
    'image_file_name': f'Example_recon_{CONFIG["recon_type"]}',
    'save_plot': False,
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


for reconstruction, samp_rat, seed in product(CONFIG['recon_type'], CONFIG['samp_rat'], CONFIG['seed']):
    week5 = {}
    CONFIG['recon_type'] = reconstruction
    CONFIG['samp_rat'] = samp_rat
    CONFIG['seed'] = seed
    results, recon = main(CONFIG, reconstruction)

    stack, A, metadata = patterns.produce_pattern(CONFIG)
    stats = matrix_stats.compute_all_stats(stack, A)
    metrics, recon = main(CONFIG, reconstruction)
    week5 = {**metadata, **stats, **metrics}
    rows.append(week5)

df = pd.DataFrame(rows)
if SAVING_INFO['save_data']:
    data = df.to_csv(f"{SAVING_INFO['data_file_name']}.csv", index=False)
