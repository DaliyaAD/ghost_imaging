#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 22:32:28 2026

Produces a stack of patterns and calculates its statistics. This can be looped
over pattern types, seeds and sampling ratios. This information can be saved
into a csv file with the relevant matrix ID for each trial.

@author: ellaward
"""
import matrix_stats
import itertools
import patterns
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

"""
Dictionary for key input values.
"""

PARAM_VALUES = {
    "Binary": [20, 40, 60, 80],
    "Gaussian": ["low", "medium", "high"],
    "Correlated Noise": ["small", "moderate", "large"],
    "Hadamard": ["random", "low", "high"],
}

CONFIG = {
    'arr_size': 64,
    'seed': 48,
    'samp_rat': 80,
    'phantom_shape': 'Shepp-Logan',
    'pattern_type': 'Correlated Noise',
    'parameter_value': ['small', 'moderate', 'large'],
    'recon_type': 'CGI'
}

SAVING_INFO = {
    'save_data': True,
    'data_file_name': 'cgi_corr_stats',
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
for pattern_type in PARAM_VALUES.keys():
    for parameter_value in PARAM_VALUES[pattern_type]:
        CONFIG.update({'pattern_type': pattern_type,
                       'parameter_value': parameter_value})
        stack, A, metadata = patterns.produce_pattern(CONFIG)
        stats = matrix_stats.compute_all_stats(stack, A)
        row = {**metadata, **stats}
        rows.append(row)

df = pd.DataFrame(rows)
if SAVING_INFO['save_data']:
    df.to_csv(f'{SAVING_INFO["data_file_name"]}.csv', index=False)
table = tabulate(df, tablefmt="grid")
# print(table)


"""
for parameter_value in PARAM_VALUES['Hadamard']:
    rows = []
    CONFIG['parameter_value'] = parameter_value
    image_title = f'Sequency={parameter_value}'
    file_name = f'Hadamard_{parameter_value}'
    stack, A, metadata = patterns.produce_pattern(CONFIG)
    disp_pattern(stack[3], image_title, file_name)
    stats = matrix_stats.compute_all_stats(stack, A)
    row = {**metadata, **stats}
    rows.append(row)

    df = pd.DataFrame(rows)
    if SAVING_INFO['save_data']:
        df.to_csv(f'{SAVING_INFO["data_file_name"]}.csv', index=False)
    table = (tabulate(df, tablefmt="grid"))
    print(table)
"""
