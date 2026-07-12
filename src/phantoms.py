#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:04:27 2026

Contains functions for phantom images shapes with defineable shape parameters. 
Grid set up allows mroe complex phantoms to be made.
make_phantom function defines the selection process for the phantom image.

@author: ellaward
"""

import numpy as np
import matplotlib.pyplot as plt


def grid_size(arr_size):
    """
    Defines the arr_size x arr_size as a co-ordinate system for more complex phantoms

    Returns
    -------
    x : 1D array, x-coordinate
    y : 1D array, y-coordinate
    """
    y, x = np.meshgrid(np.arange(arr_size), np.arange(arr_size), indexing='ij')
    return x, y


def array_sq(width, arr_size):
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
    arr_sq = np.zeros((arr_size, arr_size))
    arr_sq[arr_size//2-half_width:arr_size//2+half_width,
           arr_size//2-half_width:arr_size//2+half_width] = 1
    return arr_sq


def array_circ(radius, arr_size):
    """
    Circle phantom

    Parameters
    ----------
    radius : integer, desired circle radius

    Returns
    -------
    arr_circ : 2D array, displayed as a circle

    """
    x, y = grid_size(arr_size)
    circle = np.sqrt((x-arr_size//2)**2+(y-arr_size//2)**2)
    arr_circ = (circle <= radius).astype(int)
    return arr_circ


def array_ring(radius, arr_size):
    """
    Ring phantom

    Parameters
    ----------
    radius : integer, desired ring radius

    Returns
    -------
    arr_ring : 2D array, displayed as a ring

    """
    x, y = grid_size(arr_size)
    circle = np.sqrt((x-arr_size//2)**2+(y-arr_size//2)**2)
    arr_ring = ((radius-1 <= circle) & (circle <= radius+1)).astype(int)
    return arr_ring


def array_elp(maj_ax, min_ax, arr_size):
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
    x, y = grid_size(arr_size)
    ellipse = np.sqrt(((x-arr_size//2)/min_ax)**2+((y-arr_size//2)/maj_ax)**2)
    arr_elp = (ellipse <= 1).astype(int)
    return arr_elp


def array_sin(amplitude, frequency, thickness, arr_size):
    """
    Sinusoidal wave phantom

    Parameters
    ----------
    amplitude : integer, desired wave amplitude
    frequency : integer, desired wave frequency (recommend 0.2-0.6)
    thickness : integer, desired wave thickness (recommend 1 or 2)
    offset : integer, the default is arr_size//2.

    Returns
    -------
    arr_sin : 2D array, displays sine wave

    """
    x, y = grid_size(arr_size)
    sin_wave = amplitude*np.sin(x*frequency)+(arr_size//2)
    arr_sin = (np.abs(y - sin_wave) <= thickness).astype(int)
    return arr_sin


def make_letter_array(letter, arr_size, font_size=None, font_weight='bold',
                      threshold=0.5):
    """
    Rasterizes a single letter into a binary 2D array.

    Parameters
    ----------
    letter : str, single character to render (e.g. 'A')
    arr_size : int, size of the output array (arr_size x arr_size)
    font_size : float, font size in points. Defaults to ~0.7 * arr_size.
    font_weight : str, matplotlib font weight (e.g. 'bold', 'normal')
    threshold : float, cutoff (0-1) for binarizing the rendered greyscale image

    Returns
    -------
    letter_arr : 2D numpy array, binary (0/1) image of the letter
    """
    font_size = arr_size * 0.6

    dpi = 100
    fig_inches = arr_size / dpi
    fig = plt.figure(figsize=(fig_inches, fig_inches), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ax.text(0.5, 0.45, letter, fontsize=font_size, fontweight=font_weight,
            ha='center', va='center', color='black')

    fig.canvas.draw()
    # Convert rendered figure to a greyscale numpy array
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    grey = img[:, :, :3].mean(axis=2) / 255.0  # 0 = black text, 1 = white bg
    letter_arr = (grey < threshold).astype(int)  # 1 where letter is drawn

    return letter_arr

# PHANTOM IMAGE CHOICE


def make_phantom(shape, arr_size, width=10, radius=10, maj_ax=None, min_ax=None,
                 amplitude=None, frequency=None, thickness=None):
    """
    Defines the phantom image by selecting a shape and its parameters.

    Parameters
    ----------
    shape : str, one of 'square', 'circle', 'ring', 'ellipse', 'sine'
    width : integer, required for 'square'
    radius : integer, required for 'circle' or 'ring'
    maj_ax, min_ax : integers, required for 'ellipse'
    amplitude, frequency, thickness : numbers, required for 'sine'
    offset : integer, optional for 'sine', defaults to arr_size//2

    Returns
    -------
    2D array, the phantom pattern
    """
    if shape == "square":
        return array_sq(width, arr_size)
    elif shape == "circle":
        return array_circ(radius, arr_size)
    elif shape == "ring":
        return array_ring(radius, arr_size)
    elif shape == "ellipse":
        return array_elp(maj_ax, min_ax, arr_size)
    elif shape == "sine":
        return array_sin(amplitude, frequency, thickness, arr_size)
    elif shape == "A":
        return make_letter_array("A", arr_size)
    else:
        raise ValueError(f"Unknown phantom shape '{shape}'. "
                         f"Choose from 'square', 'circle', 'ring', 'ellipse', 'sine'.")
