#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:21:08 2026

@author: ellaward
"""
"""
take one 2D mask ->  
subtract its mean -> 
compute autocorrelation using FFT -> 
normalise by central value -> 
take horizontal and vertical centre profiles -> 
find where profile falls below 0.5 -> 
average horizontal and vertical FWHM -> 
average over all M patterns.
"""


from patterns import gen_noise
from scipy.fft import ifft2
from scipy.fft import fft2
from scipy.fft import fftshift
import numpy as np
CONFIG = {
    'arr_size': 32,
    'seed': 48,
    'samp_rat': 50,
    'parameter_value': 'moderate',
}


"""
The Wiener-Khinchin theorum is used to determine the autocorrelation function.
The theorum says that the power spectral density is equal to the fourier 
transform of a process' autocorrelation function.
i.e:
The power spectral density is defined as the Fourier transform of the function
multiplied by its complex conjugate. 
The autocorrelation is the inverse of this power spectrum.
The 'profiles' are the centre of the autocorrelation from which, I can 
nucleate an area within a certain tolerance to quantify the audocorrelation width.

To find a FWHM, start from the central profile, then step outwards until the tolerance is exceeded
Taking the average in both directions gives the autocorrelation width
"""


def fwhm(profile, centre):
    right = centre
    while right < len(profile) - 1 and profile[right] >= 0.5:
        right += 1

    left = centre
    while left > 0 and profile[left] >= 0.5:
        left -= 1
    return right - left


def autocorrelation(stack):
    slice_fwhms = []
    M = len(stack)
    for i in range(M):
        slice_2d = stack[i]
        mean = np.mean(slice_2d)
        subt_mean = slice_2d - mean
        F = fft2(subt_mean)
        power_spec = F * np.conj(F)
        # The real part of the 2D inverse Fourier transform gives autocorrelation
        autocorr = ifft2(power_spec).real
        autocorr = fftshift(autocorr)  # Peak value is shifted to the centre
        centre_y, centre_x = np.array(autocorr.shape) // 2
        autocorr = autocorr / autocorr[centre_y, centre_x]
        # PROFILES
        h_profile = autocorr[centre_y, :]
        v_profile = autocorr[:, centre_x]
        # FWHM
        h_fwhm = fwhm(h_profile, centre_x)
        v_fwhm = fwhm(v_profile, centre_y)
        avg_fwhm = (h_fwhm + v_fwhm) // 2
        slice_fwhms.append(avg_fwhm)
    stack_fwhm = np.mean(slice_fwhms)
    print(stack_fwhm)
    return stack_fwhm


patterns = gen_noise(CONFIG)
autocorrelation(patterns)
