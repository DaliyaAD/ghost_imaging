#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:43:50 2026

@author: ellaward
"""

import numpy as np
import matplotlib.pyplot as plt

SIZE = 32
CENTRE = SIZE//2
UPPER_M = 4000
INTERVAL = 250


def grid_size():
    """
    Defines the SIZE x SIZE as a co-ordinate system for more complex phantoms

    Returns
    -------
    x : 1D array, x-coordinate
    y : 1D array, y-coordinate
    """
    y, x = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing='ij')
    return x, y


def array_sq(width):
    """
    Square phantom

    Parameters
    ----------
    width : integer, desired width of square.

    Returns
    -------
    arr_sq : 2D array, displayed as a square

    """
    half_width = width//2
    arr_sq = np.zeros((SIZE, SIZE))
    arr_sq[CENTRE-half_width:CENTRE+half_width,
           CENTRE-half_width:CENTRE+half_width] = 1
    return arr_sq


def array_circ(diameter):
    """
    Circle phantom

    Parameters
    ----------
    diameter : integer, desired circle radius

    Returns
    -------
    arr_circ : 2D array, displayed as a circle

    """
    x, y = grid_size()
    circle = np.sqrt((x-CENTRE)**2+(y-CENTRE)**2)
    arr_circ = (circle <= diameter).astype(int)
    return arr_circ


def array_ring(diameter):
    """
    Ring phantom

    Parameters
    ----------
    diameter : integer, desired ring diameter

    Returns
    -------
    arr_ring : 2D array, displayed as a ring

    """
    x, y = grid_size()
    circle = np.sqrt((x-CENTRE)**2+(y-CENTRE)**2)
    arr_ring = ((diameter-1 <= circle) & (circle <= diameter+1)).astype(int)
    return arr_ring


def array_elp(maj_ax, min_ax):
    """
    Ellipse phantom

    Parameters
    ----------
    maj_ax : integer, desired ellipse major axis
    min_ax : integer, desired ellipse minor axis

    Returns
    -------
    arr_elp : 2D array, displayed as an ellipse

    """
    x, y = grid_size()
    ellipse = np.sqrt(((x-CENTRE)/min_ax)**2+((y-CENTRE)/maj_ax)**2)
    arr_elp = (ellipse <= 1).astype(int)
    return arr_elp


def array_sin(amplitude, frequency, thickness, offset=CENTRE):
    """
    Sinusoidal wave phantom

    Parameters
    ----------
    amplitude : integer, desired wave amplitude
    frequency : integer, desired wave frequency (recommend 0.2-0.6)
    thickness : integer, desired wave thickness (recommend 1 or 2)
    offset : integer, the default is CENTRE.

    Returns
    -------
    arr_sin : 2D array, displays sine wave

    """
    x, y = grid_size()
    sin_wave = amplitude*np.sin(x*frequency)+offset
    arr_sin = (np.abs(y - sin_wave) <= thickness).astype(int)
    return arr_sin


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
    #plt.savefig(file_name, dpi=400)
    plt.show()


# PHANTOM IMAGE CHOICE
PATTERN = array_sin(8, 0.3, 2)

# PLOTTING INFORMATION
PLOT_TITLE = 'The Effect of M Value on Image Reconstruction Quality'
X_LABEL = 'M value'
Y_LABEL = 'Deviation'
AUTO_X_LIMITS = True
X_LIMITS = [0., 10.]  # Not used unless AUTO_X_LIMITS = False
AUTO_Y_LIMITS = True
Y_LIMITS = [0., 10.]  # Not used unless AUTO_Y_LIMITS = False
LINE_COLOUR = 'red'
LINE_STYLE = '-'
MARKER_STYLE = 'o'
MARKER_COLOUR = 'black'
GRID_LINES = True
SAVE_FIGURE = True
FIGURE_NAME = 'BucketDetector.png'
FIGURE_RESOLUTION = 400


def create_plot(x_data, y_data, file_name):
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
    axes_main_plot.xaxis.set_major_locator(plt.MultipleLocator(INTERVAL*2))
    axes_main_plot.scatter(x_data, y_data)
    axes_main_plot.set_title(PLOT_TITLE, fontsize=14)
    axes_main_plot.set_xlabel(X_LABEL, fontsize=18)
    axes_main_plot.set_ylabel(Y_LABEL, fontsize=18)
    axes_main_plot.tick_params(axis='both', labelsize=12)
    plt.savefig(file_name, dpi=400)


def array_rand():
    """
    Create a SIZE x SIZE array of randomly generated pixels (0 or 1)

    Returns
    -------
    arr_rand : 2D array

    """
    arr_rand = np.random.randint(0, 2, size=(SIZE, SIZE))
    return arr_rand


def many_array_rand(M):
    """
    Stacks M random 2D patterns into a 3D array

    Returns
    -------
    many_rand : 3D array
    """
    many_rand = np.array([array_rand() for i in range(M)])
    return many_rand


def rand_phant_prod(random, phantom):
    """
    Takes the product of two N-D arrays. Used in order to demonstrate how 'close'
    a random pattern is to the phantom.

    Parameters
    ----------
    random : N-D array, random pattern
    phantom : N-D array, phantom image

    Returns
    -------
    prod : N-D array
    """
    prod = random*phantom
    return prod


def reconstruct_image(var_1, var_2, M):
    """
    Takes the covariance between two variables.
    Using this for the stack of random 2D pattern arrays and their respective
    weightings/bucket values, this function will display the reconstruction 
    image.

    Parameters
    ----------
    var_1 : array, first variable (random stack)
    var_2 : array, second variable (bucket values)

    Returns
    -------
    reconstructed_image : 2D array, displayable reconstruction of phantom

    """
    weighted_12 = np.zeros((M, SIZE, SIZE))
    for i in range(M):
        weighted_12[i] = var_1[i]*var_2[i]  # This is 3D
    mean_weighted_12 = np.mean(weighted_12, axis=0)  # This is 2D
    mean_1 = np.mean(var_1, axis=0)
    mean_2 = np.mean(var_2, axis=0)  # This is 2D
    reconstructed_image = (mean_weighted_12 -
                           (mean_1*mean_2))
    return reconstructed_image


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


def iterate():
    """
    This function iterates over interval values of M to determine the deviation
    and display the reconstructed pattern.

    Returns
    -------
    M_values : array, number of random patterns used.
    dev_values : array, the deviation from the phantom image for each M
    """
    M_list = list(range(INTERVAL, UPPER_M+1, INTERVAL))
    dev_values = np.zeros(len(M_list))
    for index, M in enumerate(M_list):
        BUCKET_VALUES = np.zeros(M)
        STACK = many_array_rand(M)
        for i in range(M):
            bucket = rand_phant_prod(STACK[i], PATTERN)
            weight = np.sum(bucket)
            BUCKET_VALUES[i] = weight
        recon_image = scale_normalize(
            reconstruct_image(BUCKET_VALUES, STACK, M))
        disp_pattern(recon_image, M)
        dev_values[index] = deviation(recon_image, PATTERN)
    M_values = np.array(M_list)
    return M_values, dev_values


def main():
    """
    Displays the reconstructed images for different M values and plots the
    M values against their deviation from the phantom.
    """
    data_points = iterate()
    create_plot(data_points[0], data_points[1], "Effect_of_M.png")


main()
