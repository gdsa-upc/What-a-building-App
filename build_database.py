# -*- coding: utf-8 -*-



##import cv2 #Llibreria: http://opencv.org/
#import numpy as np # Llirebria: http://www.numpy.org/
#import matplotlib.pyplot as plt
import os
#from scipy.misc import imsave
#from scipy import misc
#from glob import glob


# Lee las imagenes de una carpeta y almacena el las ID's de cada imagen en un fichero txt

#450 imagenes en la carpeta train
#180 imagenes en la carpeta val

#num_picture_train=450+1;
#num_picture_val=180+1;

#Para utilizar el patch se debe utilizar una estructura asi:

#         /     /TerrassaBuildings900 -> Aqui estaran las imagenes
#         /     /src                  -> Aqui guardaremos los archivos .py
#         /     /txt                  -> Guardaremos los ficheros txt generados
#         /     /build_database.py


patch_imagenes_train = "./TerrassaBuildings900/train/images/"
patch_imagenes_val = "./TerrassaBuildings900/val/images/"
#dir_src = "./src"
#dir_actual = "./"
dir_archivos_txt = "./txt/"



def build_database(directorio_imagenes, directorio_txt, trainorval): #Se define una funcion
    imagenes_dir = os.listdir(directorio_imagenes) #Asigna el nombre de cada imagen a una posición del vector image_files[1]
    print("Archivos leidos del directorio(patch absoluto): " + os.path.abspath(directorio_imagenes))
    fichero_txt = open(directorio_txt + 'ID_images_'+trainorval+'.txt', 'w') # Abre el fichero donde guardaremos las ID's
    for imagenes  in imagenes_dir: #Mientras haya imagenes en el directorio imagenes_dir...
        fichero_txt.write(imagenes[0:-4] + "\n") #escribe la el nombre de la id sin el jpg y añade un INTRO al final de cada nombre



# Crear los dos ficheros txt con las ID para cada directorio

build_database(patch_imagenes_train,dir_archivos_txt,"train")
build_database(patch_imagenes_val,dir_archivos_txt,"val")
