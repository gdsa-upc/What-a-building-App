# -*- coding: utf-8 -*-

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



def classify(features_file, results_dir, vector_etiquetes):
    ImID = pickle.load( open("../descriptores/" + features_file, "r" ) ) #Obrim l'arxiu amb les Im ID'
    classi = open(result_dir + "classificacio.txt") #Creem l'arxiu per guardar els resultats
    
    #For per asignar un edifici random despres de cada id després de una tabulació: 29796-14601-23662	catedral
    for keyval, valuesval in ImID.items():
        descrip = pickle.load(open(dir_descriptores+features_file, "r")) #Obrim l'arxiu per llegir les im ID's
        classi.write(keyval + "/t" + vector_etiquetes[np.random.randint(0, 12)]) #Asignem una classificació aleatoria depenent de la posició (random) del vector d'etiquetes
        descrip.close()
        
    classi.close()
   
#definim un vector amb totes les posibles etiquetes:

vector_etiquetes = ["mnactec", "mercat_independencia", "ajuntament", "societat_general", "estacio_nord", "dona_treballadora", "escola_enginyeria", "catedral", "teatre_principal", "farmacia_albinyana", "masia_freixa", "castell_cartoixa", "desconegut"]


#cridem a la funció:

classify ("descriptor_train.p", dir_archivos_txt, vector_etiquetes)
