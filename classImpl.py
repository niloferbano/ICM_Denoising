#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 13:16:19 2018

@author: nilu
"""

from scipy import misc
import numpy as np

import matplotlib.pyplot as plt

class ICM:
    def __init__(self, filename, h=0 , beta=1.0, eta=2.1):
        """
        Initialize the image variale with noise image and sets the hyperparameters

		"""
        self.binary_pixel = np.array([1,-1])
        self.image = misc.imread(filename)
        self.image = np.asarray(self.image)
        self.image = np.where(self.image<255,-1,1)
        self.initialize_hyperparameter(h, beta, eta)
        self.xrange = self.image.shape[0]
        self.yrange = self.image.shape[1]
        plt.imshow(self.image , cmap = 'gray')
        plt.show()

    def add_noise(self,noise = 0):
        if noise == 0:
            return self.image
        else:
            self.__init__("Download-Free-Binary-Code-Wallpaper_binary")
            noise_rand = np.random.normal(self.image.shape[0],self.image.shape[1])
            noise_image_ = np.where(noise_rand<noise,-1,1)
            noisy_image = self.image * noise_image_
            return noisy_image
        
    def initialize_hyperparameter(self,h, beta, eta):
        self.h = h
        self.beta = beta
        self.eta = eta

    def apply_ICM(self, noise = 0):
        self.noisy_image = self.add_noise(noise)
        [self.update_pixel(i,j) for k in (0,2) for i in range(0, self.noisy_image.shape[0]) for j in range(0,self.noisy_image.shape[1])]
    
    def update_pixel(self,i, j):
        self.noisy_image_ref = self.noisy_image
 
        latent_factor = 0
        neighbour_factor = 0
        bias = 0
        
        x_i = self.binary_pixel[0] # for pixel value +1
        y_i =self. noisy_image_ref[i][j]
        if i-1 >= 0:
           latent_factor += self.noisy_image[i-1][j]*x_i
        if i+1 < self.xrange:
            latent_factor += self.noisy_image[i+1][j]*x_i
        if j-1 >= 0:
            latent_factor += self.noisy_image[i][j-1]*x_i
        if j+1 < self.yrange:
            latent_factor += self.noisy_image[i][j+1]*x_i
        neighbour_factor = x_i*y_i
        bias = x_i
        energy_plus1 = self.h*bias - self.beta*latent_factor - self.eta*neighbour_factor
        
        x_i = self.binary_pixel[1] # for pixel value -1
        bias = 0
        latent_factor = 0
        neighbour_factor = 0

        if i-1 >= 0:
           latent_factor +=self. noisy_image[i-1][j]*x_i
        if i+1 < self.xrange:
            latent_factor += self.noisy_image[i+1][j]*x_i
        if j-1 >= 0:
            latent_factor += self.noisy_image[i][j-1]*x_i
        if j+1 < self.yrange:
            latent_factor += self.noisy_image[i][j+1]*x_i   
        neighbour_factor = x_i*y_i
        bias = x_i
        energy_minus1 = self.h*bias - self.beta*latent_factor - self.eta*neighbour_factor
        
        if energy_plus1 < energy_minus1:
            self.noisy_image[i][j] = 1
        else:
            self.noisy_image[i][j] = -1
        self.noisy_image_ref = self.noisy_image

                          
icm = ICM('Download-Free-Binary-Code-Wallpaper_noisy.bmp', 0 , 1.0, 2.1)
icm.apply_ICM();
plt.imshow(icm.noisy_image_ref, cmap = 'gray')
plt.show()