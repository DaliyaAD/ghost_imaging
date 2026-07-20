#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:35:14 2026

@author: ellaward
"""
import patterns
from tabulate import tabulate
import pandas as pd

CONFIG = {
    'arr_size': 32,
    'seed': 43,
    'M': 4,
    'pattern_type': 'hadamard',
    'grain_size': 'large',
    'gaussian_contrast': 'high',
    'binary_density': 40
}


def save_stats(CONFIG):
    stats, pattern = patterns.produce_pattern(CONFIG)
    stats_names = ['pattern_type', 'parameter_name', 'parameter_value',
                   'M', 'arr_size', 'seed', 'mean_intensity', 'variance']
    df = pd.DataFrame([stats], columns=stats_names)
    data = df.to_csv("matrix_stats.csv", index=False)


save_stats(CONFIG)
