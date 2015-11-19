# -*- coding: utf-8 -*-


#Aquest script ha de“()

import cv2 #Llibreria: http://opencv.org/
import numpy as np # Llirebria: http://www.numpy.org/
import matplotlib.pyplot as plt
from scipy.misc import imsave
from scipy import misc
from glob import glob



image_data = np.zeros(512*512, dtype=np.float32).reshape(512,512); #Crea un array(matriz) de zeros tipo float
random_data = np.random.randn(512,512) ;#Te genera un array(matriz) con numeros random de 512*512
random_data[1][4]; #acceder a la posicion 1,4 del array.
print 'Size: ', image_data.size  #tamaño
print 'Shape: ', image_data.shape #forma 512x512

scaled_image_data = image_data / 255
imsave('noise.png', scaled_image_data)

## información útil: http://prancer.physics.louisville.edu/astrowiki/index.php/Image_processing_with_Python_and_SciPy


#Te genera 10 imagenes .png con valores random de 0  a 255 y le hace un rescalado a 100x100
for i in range(10):
    #im = np.random.randn(512,512)
    im = np.random.random_integers(0, 255, 10000).reshape((100, 100))
    misc.imsave('random_%02d.png' % i, im)

filelist = glob('random*.png')
filelist.sort()

im[1] #imagen 1 creada..
im[3].size
im[4].shape
