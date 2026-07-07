#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:49:17 2026

@author: ellaward
"""

from CGI_sim_week_2 import iterate
from CGI_sim_week_2 import make_phantom
from tabulate import tabulate


ARR_SIZE = 32
SHAPE, SHAPE_PARAMS = "ring", {"radius": 10}
INTERVAL, UPPER_M = 750, 3000
SEED = 43

phantom = make_phantom(SHAPE, ARR_SIZE, **SHAPE_PARAMS)
data = iterate(phantom, ARR_SIZE, interval=INTERVAL,
               upper_m=UPPER_M, arr_seed=SEED)

# Make table
headers = ["Sampling ratio", "NMSE", "PSNR", "SSIM"]
rows = list(zip(data[1], data[3], data[4], data[5]))
print(tabulate(rows, headers=headers, tablefmt="grid"))
