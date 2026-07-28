#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:56:15 2026

@author: ellaward
"""
"""
The sensing matrix has dimensions (M, N), where N = arr_size**2.
The rows of this matrix give the flattened versions of the individual patterns.
The collumns give how the pixel value changes across M patterns.

Mutual coherence is a measure of how correlated these collumns are.
We can calculate mutual coherence using the maximum, absolute, normalised inner 
product between the collumns.

"""


import numpy as np
from patterns import gen_noise
from patterns import gen_matrix_A
CONFIG = {
    'arr_size': 32,
    'seed': 48,
    'samp_rat': 50,
    'parameter_value': 'moderate',
}

pattern = gen_noise(CONFIG)
A_patt = gen_matrix_A(pattern)

# NORMALIZATION
norms = np.linalg.norm(A_patt, axis=0, keepdims=True)
normalization = A_patt / norms

# INNER PRODUCT
"""
The inner product between columns of a sensing matrix is called a Gram matrix.
It is calculated as G = A*A. The off-diagonal entries of the Gram matrix 
describe the correlation between the columns.
If the columns are linearly independent, the Gram determinant will take a value
of zero.
Simply using the numpy function .T will give the transpose of the matrix. As we
can be sure all elements will contain real values, G = A.T @ A, where @ is the
operator for matrix multiplication
"""
gram_matrix = normalization.T @ normalization

# MAX VALUE
"""
The diagonal entries in the Gram matrix are the squared lengths of the collumns.
These will be the largest value in matrix. When deriving the collumn coherence,
we have to negate these diagonal values.
"""
np.fill_diagonal(
    gram_matrix, 0)  # Don't assign to a variable as this function directly modifies gram_matrix.
mutual_coherence = np.max(gram_matrix)
# print(mutual_coherence)

# SIGNIFICANCE
"""
A low mutual coherence means that the pixels are sampled more independently.
The lack of relation between collumns will produce a clearer reconstruction 
without the build up from correlated noise.
'Correct' pixels will be reinforced more strongly than 'incorrect' pixels as 
they are more distinct.
"""
