# -*- coding: utf-8 -*-
import cv2
import pickle #Ejemplos de serialización: https://docs.python.org/2/library/pickle.html
import numpy as np
import os

#450 imagenes en la carpeta train
#180 imagenes en la carpeta val

#num_picture_train=450+1;
#num_picture_val=180+1;

#    Para utilizar el path se debe utilizar una estructura asi:

#    TerrassaBuildings900               -> Aquí deben estar las imágenes
#    src                                -> Aquí deben estar los archivos .py
#    txt                                -> Aquí guardaremos los ficheros txt generados

path_imagenes_train = "../TerrassaBuildings900/train/images/"
path_imagenes_val = "../TerrassaBuildings900/val/images/"
dir_archivos_txt = "../txt/"
dir_descriptores = "../descriptores/"

#Asignamos un valor aleatorio de clasificacion para cada imagen de val/test:


def get_features(feature_file, direc_guardat, possible_labels):
    archivo_imageid = open(dir_archivos_txt+archivo_image_id, 'r') #obrim l'arxiu amb les id de les imatges
    
    if not os.path.exists(directorio_descriptores): #ens asegurem de que el directori sigui el correcte
        os.makedirs(directorio_descriptores)
    
  
    for id_linia in archivo_imageid: #engeguem el for per a implementar la classificació aleatòria.
        vector = []
        #aqui es donde me falla el cerebro.
