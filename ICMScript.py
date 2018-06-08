#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 13:18:25 2018

@author: nilu
"""

from scipy import misc
import numpy as np

import matplotlib.pyplot as plt


im = misc.imread('Download-Free-Binary-Code-Wallpaper_noisy.bmp')
im = np.asarray(im)
imgplot = plt.imshow(im, cmap = 'gray')
plt.show()
im = np.where(im<255,-1,1)

noisy_image_ref = noisy_image = im

binary_pixel = np.array([1,-1])

h = 1
beta = 1.0
eta = 2.1
def add_noise(noise = 0):
    global noisy_image_ref
    global noisy_image
    if(noise == 0):
        return
    im = misc.imread('Download-Free-Binary-Code-Wallpaper_binary.bmp')
    im = np.where(im<255,-1,1)
    noise_im = np.random.normal(im.shape[0],im.shape[1])
    noisy = np.where(noise_im<noise,-1,1)
    noisy_image = im * noisy
    noisy_image_ref = noisy_image
    
def ICM(i, j):
    global noisy_image_ref
    global noisy_image
    latent_factor = 0
    neighbour_factor = 0 
    bias = 0 
    x_i = binary_pixel[0] # for binary pixel value +1
    y_i = noisy_image_ref[i][j]
    if i-1 >= 0:
        latent_factor += noisy_image[i-1][j]*x_i
    if i+1 < noisy_image_ref.shape[0]:
        latent_factor += noisy_image[i+1][j]*x_i
    if j-1 >= 0:
        latent_factor += noisy_image[i][j-1]*x_i
    if j+1 < noisy_image_ref.shape[1]:
        latent_factor += noisy_image[i][j+1]*x_i
    neighbour_factor = x_i*y_i
    bias = x_i
    energy_plus1 = h*bias - beta*latent_factor - eta*neighbour_factor
    bias = 0
    latent_factor = 0
    neighbour_factor = 0
    x_i = binary_pixel[1] # for binary pixel value -1
    if i-1 >= 0:
        latent_factor += noisy_image[i-1][j]*x_i
    if i+1 < noisy_image_ref.shape[0]:
        latent_factor += noisy_image[i+1][j]*x_i
    if j-1 >= 0:
        latent_factor += noisy_image[i][j-1]*x_i
    if j+1 < noisy_image_ref.shape[1]:
        latent_factor += noisy_image[i][j+1]*x_i
    neighbour_factor = x_i*y_i
    bias = x_i
    energy_minus1 =  h*bias - beta*latent_factor - eta*neighbour_factor
    if energy_plus1 < energy_minus1:
        noisy_image[i][j] = 1
    else:
        noisy_image[i][j] = -1
    noisy_image_ref = noisy_image
    
# ICM
add_noise(0.2)
[ICM(i,j) for k in (0,2) for i in range(0, noisy_image.shape[0]) for j in range(0, noisy_image.shape[1])]
	
plt.imshow(noisy_image_ref, cmap = 'gray')
plt.show()