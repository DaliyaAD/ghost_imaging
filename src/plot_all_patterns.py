#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 23:43:23 2026

@author: ellaward
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


df = pd.read_csv('cgi_corr_stats.csv')

metrics = {
    'contrast': 'Contrast',
    'entropy': 'Entropy',
    'autocorrelation width': 'Autocorrelation width',
    'effective_rank': 'Effective rank',
    'mutual coherence': 'Mutual coherence'
}

for col, ylabel in metrics.items():
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.rcParams['font.family'] = 'Times New Roman'

    sns.stripplot(
        data=df,
        x='pattern_type',
        y=col,
        hue='parameter_value',
        dodge=True,        # <-- separates parameter values within each pattern family
        jitter=True,        # spreads overlapping points sideways so they don't stack exactly
        size=8,
        edgecolor='black',
        linewidth=0.5,
        alpha=0.8,
        ax=ax
    )

    ax.set_xlabel('Pattern family', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(
        f'{ylabel} by pattern family and parameter value', fontsize=14)
    ax.legend(title='Parameter value')

    fig.tight_layout()
    safe_name = col.replace(' ', '_')
    fig.savefig(f'{safe_name}_scatter_by_pattern_family.png', dpi=300)
    plt.show()
