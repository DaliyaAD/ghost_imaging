#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:35:14 2026

@author: ellaward
"""
import patterns_mat_A
import numpy as np
from tabulate import tabulate
import pandas as pd
from skimage.measure import shannon_entropy as ent
import matplotlib.pyplot as plt
import itertools
eps = 1e-12


def calc_pattern_contrast(array):
    pattern_contrast = []
    for i in range(len(array)):
        slice_2d = array[i]
        contrast = np.std(slice_2d)/(np.mean(slice_2d) + eps)
        pattern_contrast.append(contrast)
    return pattern_contrast


def calc_pattern_entropy(array):
    pattern_entropy = []
    for i in range(len(array)):
        slice_2d = array[i]
        entropy = ent(slice_2d)
        pattern_entropy.append(entropy)
    return pattern_entropy


def stack_statistics(stack):
    return {
        "mean": float(np.mean(stack)),
        "var": float(np.var(stack)),
        "std": float(np.std(stack)),
        "contrast": float(np.std(stack)/(np.mean(stack) + eps)),
        "entropy": float(np.mean(calc_pattern_entropy(stack)))
    }


def compute_matrix_stats(A):
    rank = np.linalg.matrix_rank(A)
    singular_values = np.linalg.svd(A, compute_uv=False)
    cond_number = singular_values[0] / \
        singular_values[-1] if singular_values[-1] > 0 else np.inf

    return {
        "effective_rank": int(rank),
        "cond_number": float(cond_number),
    }


def compute_all_stats(stack, A):
    stats = {}
    stats.update(stack_statistics(stack))
    stats.update(compute_matrix_stats(A))
    return stats
