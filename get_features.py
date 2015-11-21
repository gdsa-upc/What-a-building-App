# -*- coding: utf-8 -*-

import cv2
#import cPickle as pickle
import pickle #Ejemplos de serialización: https://docs.python.org/2/library/pickle.html
import numpy as np


#450 imagenes en la carpeta train
#180 imagenes en la carpeta val

#num_picture_train=450+1;
#num_picture_val=180+1;

#Para utilizar el patch se debe utilizar una estructura asi:

#         /     /TerrassaBuildings900 -> Aqui estaran las imagenes
#         /     /src                  -> Aqui guardaremos los archivos .py
#         /     /txt                  -> Guardaremos los ficheros txt generados
#         /     /build_database.py
#         /     /get_features.py
#         /     /descriptores/train y /val




patch_imagenes_train = "./TerrassaBuildings900/train/images/"
patch_imagenes_val = "./TerrassaBuildings900/val/images/"
#dir_src = "./src"
#dir_actual = "./"
dir_archivos_txt = "./txt/"
dir_descriptores = "./descriptores/"
dir_descrip_val = "./descriptores/val/"
dir_descrip_train = "./descriptores/train/"

#archivo image id es el nombre del archivo. En este caso hay dos ID_images_train.txt o ID_images_val.txt

def get_features(directorio_imagenes, archivo_image_id, directorio_descriptores):

    archivo_imageid = open(dir_archivos_txt+archivo_image_id, 'r') #Abrir el archivo de con las ID's de las imangenes
    descriptor_file = open(directorio_descriptores+"Descriptors.p", 'w') #Abrir un fichero txt para guardar los descriptores
    dict_caracteristicas={} #creamos un diccionario vacio
    for id_linia in archivo_imageid:

        vector_caract_random = np.random.random_integers(0, 255, 100) ;#Te genera un array con  100 numeros random entre 0 y 255
        #a = id_linia.split();
        #print('Entro')
        #dict_caracteristicas = id_linia.split()
        #dict_caracteristicas = vector_caract_random  #afegim el vector de caracteristiques aleatori al diccionari
        #print(dict_caracteristicas)
        final = id_linia.index("\n") #obtenim la posició del salt de línia
        dict_caracteristicas[id_linia[0:final]] = vector_caract_random #afegim el vector de caracteristiques aleatori al diccionari

    pickle.dumps = (dict_caracteristicas, descriptor_file)
    archivo_imageid.close()
    descriptor_file.close()


get_features(patch_imagenes_train,"ID_images_train.txt", dir_descriptores)
