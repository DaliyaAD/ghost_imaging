#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 09:55:27 2026

@author: ellaward
"""


from algorithms import main
import pandas as pd
import matrix_stats
import patterns
from itertools import product

rows = []

CONFIG = {
    'arr_size': 32,
    'seed': [47],
    'samp_rat': [30, 60, 90],
    'phantom_shape': 'Shepp-Logan',
    'recon_type': ["CGI", "DGI", "Ridge"],
    'pattern_type': None,
    'parameter_value': None,

}

PARAM_VALUES = {
    "Binary": [30, 60, 90],
    "Gaussian": ["low", "medium", "high"],
    "Correlated Noise": ["small", "moderate", "large"],
    "Hadamard": ["random", "low", "high"],
}

SAVING_INFO = {
    'save_data': True,
    'data_file_name': 'dataset',
    'save_pattern_image': False,
    'image_title': None,
    'image_file_name': None,
    'save_plot': False,
    'plot_file_name': None,
    'plot_title': None,
    'x_label': 'Sampling Ratio (%)',
    'y_label': 'SSIM',
}

for pattern_type, samp_rat, seed in product(
        PARAM_VALUES.keys(),
        CONFIG['samp_rat'],
        CONFIG['seed']):

    for parameter_value in PARAM_VALUES[pattern_type]:
            config = CONFIG.copy()
            config.update({
                'pattern_type': pattern_type,
                'parameter_value': parameter_value,
                'samp_rat': samp_rat,
                'seed': seed,
            })
            stack, A, metadata = patterns.produce_pattern(config)
            stats = matrix_stats.compute_all_stats(stack, A)
            for reconstruction in config['recon_type']:
                metrics, recon = main(config, reconstruction)
                matrix_id = f"{reconstruction}_{config['pattern_type'].lower()}_{config['parameter_value']}_size{config['arr_size']}_samprat{config['samp_rat']}_seed{config['seed']}"
                POOP = {
                    "matrix_id": matrix_id,
                    "recon_type": reconstruction,
                }
                week5 = {**POOP, **metadata, **stats, **metrics}
                rows.append(week5)

df = pd.DataFrame(rows)
if SAVING_INFO['save_data']:
    data = df.to_csv(f"{SAVING_INFO['data_file_name']}.csv", index=False)
