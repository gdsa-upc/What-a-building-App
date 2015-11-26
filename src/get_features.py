# -*- coding: utf-8 -*-
import cv2
import pickle #Ejemplos de serialización: https://docs.python.org/2/library/pickle.html
import numpy as np
import os

#450 imagenes en la carpeta train
#180 imagenes en la carpeta val

#archivo image id es el nombre del archivo. En este caso hay dos ID_images_train.txt o ID_images_val.txt

def get_features(directorio_imagenes, archivo_image_id, directorio_descriptores):


    archivo_imageid = open(dir_archivos_txt+archivo_image_id, 'r') #Abrir el archivo de con las ID's de las imangenes
    dict_caracteristicas={} #creamos un diccionario vacio
    
    if not os.path.exists(directorio_descriptores):
        os.makedirs(directorio_descriptores)


    for id_linia in archivo_imageid:

        vector_caract_random = np.random.random_integers(0, 255, 100) ;#Te genera un array con  100 numeros random entre 0 y 255
        linia = id_linia.split(); # lee la linia del fichero
        dict_caracteristicas [linia[0]] = vector_caract_random #asignamos por cada linia un vector de caracteristicas diferente


        if archivo_image_id == "ID_images_train.txt":
            pickle.dump(dict_caracteristicas, open(directorio_descriptores+"descriptor_" + "train.p", "wb" ) ) # guarda el diccionario en un archivo .p
        else:
            pickle.dump(dict_caracteristicas, open(directorio_descriptores+"descriptor_" + "val.p", "wb" ) ) # guarda el diccionario en un archivo .p



    archivo_imageid.close()
    #descriptor_file.close()


if __name__=="__main__":

    #    Para utilizar el path se debe utilizar una estructura asi:

    #    TerrassaBuildings900               -> Aquí deben estar las imágenes
    #    src                                -> Aquí deben estar los archivos .py
    #    txt                                -> Aquí guardaremos los ficheros txt generados
    #    descriptores                       -> Aquí guardaremos los descriptores generados


    path_imagenes_train = "../TerrassaBuildings900/train/images/"
    path_imagenes_val = "../TerrassaBuildings900/val/images/"
    dir_archivos_txt = "../txt/"
    dir_descriptores = "../descriptores/"


    get_features(path_imagenes_train,"ID_images_train.txt", dir_descriptores)
    get_features(path_imagenes_val, "ID_images_val.txt", dir_descriptores)

    #Load a pickle file
    dict_val = pickle.load( open(dir_descriptores+"descriptor_val.p", "rb" ) ) #lee el archivo creado .p donde esta el diccionario.
    dict_train = pickle.load( open(dir_descriptores+"descriptor_train.p", "rb" ) )#
