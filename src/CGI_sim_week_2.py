#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:43:50 2026

@author: ellaward
"""

import numpy as np
import matplotlib.pyplot as plt
from phantoms import make_phantom


# PLOTTING INFORMATION
PLOT_TITLE = 'The Effect of M Value on Image Reconstruction Quality'
X_LABEL = 'M value'
Y_LABEL = 'Deviation'
AUTO_X_LIMITS = True
X_LIMITS = [0., 10.]  # Not used unless AUTO_X_LIMITS = False
AUTO_Y_LIMITS = True
Y_LIMITS = [0., 10.]  # Not used unless AUTO_Y_LIMITS = False
GRID_LINES = True
SAVE_FIGURE = True
FIGURE_NAME = 'Effect_of_M.png'
FIGURE_RESOLUTION = 400


def create_plot(x_data, y_data):
    """
    Function to plot x and y data with adjustable features

    Parameters
    ----------
    x_data : 1D array, independent variable
    y_data : 1D array, dependent variable
    file_name : string, file name must end in .filetype (e.g: file_name.png)
    """
    figure = plt.figure(figsize=(8, 9))
    plt.rcParams['font.family'] = 'Times New Roman'
    axes_main_plot = figure.add_subplot(211)
    axes_main_plot.grid(GRID_LINES)
    axes_main_plot.xaxis.set_major_locator(
        plt.MultipleLocator(np.max(x_data) * 2))
    axes_main_plot.scatter(x_data, y_data)
    axes_main_plot.set_title(PLOT_TITLE, fontsize=14)
    axes_main_plot.set_xlabel(X_LABEL, fontsize=18)
    axes_main_plot.set_ylabel(Y_LABEL, fontsize=18)
    axes_main_plot.tick_params(axis='both', labelsize=12)
    plt.savefig(FIGURE_NAME, dpi=FIGURE_RESOLUTION)


def many_array_rand(M, arr_size=None, arr_seed=None):
    """
    Create an arr_size x arr_size array of randomly generated pixels (0 or 1)

    Returns
    -------
    arr_rand : 2D array

    """
    rng = np.random.default_rng(seed=arr_seed)
    arr_rand = rng.integers(0, 2, size=(M, arr_size, arr_size))
    return arr_rand


def compute_bucket_values(stack, phantom):
    """
    Computes the product between the phantom image and the speckled pattern.

    Parameters
    ----------
    stack : array, stack of random speckled patterns
    phantom : array, image under reconstruction

    Returns
    -------
    2D array

    """
    return np.sum(stack * phantom, axis=(1, 2))


def reconstruct_image(bucket_values, stack):
    """
    Reconstructs the phantom image using covariance.  Using this for the stack 
    of random 2D pattern arrays and their respective
    weightings/bucket values, this function will display the reconstruction 
    image.

    Parameters
    ----------
    bucket_values : array, sum of the product array.
    stack : array, stack of speckled patterns
    M : integer, M-value

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    weighted = stack * bucket_values[:, None, None]
    mean_weighted = np.mean(weighted, axis=0)
    mean_bucket = np.mean(bucket_values)
    mean_stack = np.mean(stack, axis=0)
    return mean_weighted - mean_bucket * mean_stack


def scale_normalize(reconstructed_pattern):
    """
    In order to compare the reconstructed image and the original image, the
    reconstructed image needs to be rescaled/normalized to have values between
    0 and 1. This is done with min-max normalization.

    Parameters
    ----------
    reconstructed_pattern : array, unscaled reconstructed image

    Returns
    -------
    normalised_pattern : array, reconstructed image normalised between 0 and 1

    """
    normalised_pattern = (reconstructed_pattern-reconstructed_pattern.min()) / \
        (reconstructed_pattern.max()-reconstructed_pattern.min())
    return normalised_pattern


def deviation(reconstruction, true_values):
    """
    Function that calculates the MSD of the reconstructed image and the 
    original

    Parameters
    ----------
    reconstruction : array, reconstructed image
    true_values : array, original phantom image

    Returns
    -------
    mean_sq_dev : float, mean square deviation of the reconstructed image to 
    the phantom image

    """
    mean_sq_dev = np.mean((reconstruction - true_values)**2)
    return mean_sq_dev


def disp_pattern(pattern, M):
    """
    Function to display 2D arrays as a binary or grayscale image

    Parameters
    ----------
    pattern : 2D array, the 2D array that needs to be displayed

    """
    plt.imshow(pattern, cmap='gray')
    plt.title(f"M={M}")
    plt.axis('off')
    plt.savefig(f"Phantom_recon_{M}.png", dpi=400)
    plt.show()


def iterate(pattern, arr_size, interval=None, upper_m=None, arr_seed=None):
    """
    Funtion that reconstructs the phantom image at different values of M.
    This iterates through values of M at designated intervals up to a maximum value.

    Parameters
    ----------
    pattern : array
    arr_size : integer, size of the image
    interval : integer, interval between M values that the code reconstructs for.
    The default is None.
    upper_m : integer, value of M up to which the code with reconstruct the phantom image.
    The default is None.
    arr_seed : integer, seed number. The default is None.

    Returns
    -------
    dev_values : Values of deviation from the phantom for each M value

    """
    M_list = list(range(interval, upper_m + 1, interval))
    full_stack = many_array_rand(upper_m, arr_size=arr_size, arr_seed=arr_seed)
    bucket_all = compute_bucket_values(full_stack, pattern)

    dev_values = np.zeros(len(M_list))
    for index, M in enumerate(M_list):
        stack_m = full_stack[:M]
        bucket_m = bucket_all[:M]
        recon = scale_normalize(reconstruct_image(bucket_m, stack_m, M))
        disp_pattern(recon, M)
        dev_values[index] = deviation(recon, pattern)
    return np.array(M_list), dev_values


def main():
    """
    Executes reconstruction of the phantom image and plots the deviation values
    against M values

    Returns
    -------
    None.

    """
    ARR_SIZE = 32
    SHAPE, SHAPE_PARAMS = "square", {"width": 10}
    INTERVAL, UPPER_M = 500, 2000
    SEED = 43

    phantom = make_phantom(SHAPE, ARR_SIZE, **SHAPE_PARAMS)
    data_points = iterate(phantom, ARR_SIZE, interval=INTERVAL,
                          upper_m=UPPER_M, arr_seed=SEED)
    create_plot(data_points[0], data_points[1])


main()
