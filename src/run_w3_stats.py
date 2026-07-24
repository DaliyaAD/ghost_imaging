#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 22:32:28 2026

@author: ellaward
"""
import matrix_stats
import itertools
import patterns_mat_A
import pandas as pd

CONFIG = {
    'arr_size': 64,
    'seed': 48,
    'samp_rat': 200,
    'phantom_shape': 'Shepp-Logan',
    'pattern_type': 'Binary',
    'save_data': False,
    'parameter_value': 80,
    'save_pattern_fig': False
}


rows = []
pattern_types = ["Binary"]
samp_rats = [100, 200]
seeds = [1, 2]
for pattern_type, samp_rat, seed in itertools.product(pattern_types, samp_rats, seeds):
    CONFIG.update({'pattern_type': pattern_type,
                  'samp_rat': samp_rat, 'seed': seed})
    stack, A, metadata = patterns_mat_A.produce_pattern(CONFIG)
    stats = matrix_stats.compute_all_stats(stack, A)
    row = {**metadata, **stats}
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv('week3_pattern_stats.csv', index=False)
