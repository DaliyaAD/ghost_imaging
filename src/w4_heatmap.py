#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 22:32:28 2026

Produces a stack of patterns and calculates its statistics. This can be looped
over pattern types, seeds and sampling ratios. This information can be saved
into a csv file with the relevant matrix ID for each trial.

@author: ellaward
"""
import seaborn as sns
import matrix_stats
import numpy as np
import itertools
import patterns
import pandas as pd
from algorithms import main
import matplotlib.pyplot as plt
from tabulate import tabulate


"""
Dictionary for key input values.
"""

PARAM_VALUES = {
    "Binary": [50],
    "Gaussian": ["high"],
    "Correlated Noise": ["small"],
    "Hadamard": ["high"]
}

CONFIG = {
    'arr_size': 64,
    'seed': [1],
    'samp_rat': 80,
    'recon_type': ['CGI', 'DGI', 'Ridge'],
    'phantom_shape': 'Shepp-Logan',
    'pattern_type': 'w4_heatmap',
    'parameter_value': None,
}

SAVING_INFO = {
    'save_data': True,
    'data_file_name': 'Hadamard',
    'save_pattern_image': False,
    'image_title': None,
    'image_file_name':  None,
    'save_plot': False,
    'plot_file_name': 'SR_vs_SSIM_Ridge',
    'plot_title': None,
    'x_label': 'Sampling Ratio (%)',
    'y_label': 'SSIM',
}
"""
Dictionary for the respective parameter values for each pattern type.
Values can be singular or a part of a list that will be looped over.
"""


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


rows = []

for pattern_type, seed in itertools.product(PARAM_VALUES.keys(), CONFIG['seed']):
    for parameter_value in PARAM_VALUES[pattern_type]:
        CONFIG.update({'pattern_type': pattern_type,
                      'parameter_value': parameter_value, 'seed': seed})
        stack, A, metadata = patterns.produce_pattern(
            CONFIG)   # done once per pattern/seed
        stats = matrix_stats.compute_all_stats(
            stack, A)         # done once per pattern/seed

        for recon_type in CONFIG['recon_type']:
            CONFIG.update({'recon_type': recon_type})
            # only reconstruction repeats per algorithm
            metrics, recon = main(CONFIG, recon_type)
            week5 = {**metadata, **stats, **metrics}
            rows.append(week5)


headers = ["Algorithm", "Phantom", "Pattern", "Sampling Ratio (%)", "NMSE",
           "PSNR", "SSIM", "Seed", "Image Size", "Runtime"]

df = pd.DataFrame(rows)
df.to_csv(f"{SAVING_INFO['data_file_name']}.csv",
          index=False)

pivot_table = df.pivot_table(
    index='pattern_type',
    columns='recon_type',
    values='ssim',
    aggfunc='mean',
)


plt.rcParams['font.family'] = 'Times New Roman'
figure, axes = plt.subplots(figsize=(8, 6))

sns.heatmap(
    pivot_table,
    annot=True,       # write the numeric value in each cell
    fmt='.3f',        # format numbers to 3 decimal places
    # color scheme; use 'viridis_r' if plotting NMSE (lower=better, so reverse it)
    cmap='viridis',
    cbar_kws={'label': 'SSIM'},
    ax=axes
)

axes.set_title(
    f'Pattern × Algorithm Reconstruction Quality (Sampling Ratio = {CONFIG["samp_rat"]})', fontsize=14)
axes.set_xlabel('Algorithm', fontsize=14)
axes.set_ylabel('Pattern', fontsize=14)

plt.tight_layout()
plt.savefig('pattern_algorithm_heatmap.png', dpi=400)
