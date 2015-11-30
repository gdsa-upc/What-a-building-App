# -*- coding: utf-8 -*-

import os


#    Lee las imagenes de una carpeta y almacena el las ID's de cada imagen en un fichero txt


#    Para utilizar el path se debe utilizar una estructura asi:

#    TerrassaBuildings900               -> Aquí deben estar las imágenes
#    src                                -> Aquí deben estar los archivos .py
#    txt                                -> Aquí guardaremos los ficheros txt generados


#    Para utilizar las variables de path debemos ejecutar el script desde la carpeta /src.

path_imagenes_train = "../TerrassaBuildings900/train/images/" #Esta variable la utilizamos para referirnos al directorio donde extraer las imagenes.
path_imagenes_val = "../TerrassaBuildings900/val/images/" #Si estamos en el directorio /src y ejecutamos ../ volvemos al directorio el cual pertenece que es nuestro directorio master.
dir_archivos_txt = "../txt/"


def build_database(directorio_imagenes, directorio_txt, trainorval):
    imagenes_dir = os.listdir(directorio_imagenes) #Asigna el nombre de cada imagen a una posición del vector image_files[1]
    print("Archivos leidos del directorio(path absoluto): " + os.path.abspath(directorio_imagenes))
    if not os.path.exists(dir_archivos_txt):
        os.makedirs(dir_archivos_txt)

    fichero_txt = open(directorio_txt + 'ID_images_'+trainorval+'.txt', 'w') # Abre el fichero donde guardaremos las ID's
    for imagenes  in imagenes_dir: # Mientras haya imagenes en el directorio imagenes_dir...
        fichero_txt.write(imagenes[0:-4] + "\n") # escribe la el nombre de la id sin el jpg y añade un INTRO al final de cada nombre




# Crear los dos ficheros txt con las ID para cada directorio

build_database(path_imagenes_train, dir_archivos_txt, 'train')
build_database(path_imagenes_val, dir_archivos_txt, 'val')

