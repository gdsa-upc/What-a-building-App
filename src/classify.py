# -*- coding: utf-8 -*-

import pickle #Ejemplos de serialización: https://docs.python.org/2/library/pickle.html
import numpy as np
import os

#    Para utilizar el path se debe utilizar una estructura asi:

#    src                                -> Aquí deben estar los archivos .py
#    txt                                -> Aquí guardaremos los ficheros txt generados

path_imagenes_train = "../TerrassaBuildings900/train/images/"
path_imagenes_val = "../TerrassaBuildings900/val/images/"
dir_archivos_txt = "../txt/"
dir_descriptores = "../descriptores/"



def classify(features_file, results_dir, vector_etiquetes):
    ImID = pickle.load( open(dir_descriptores + features_file, "r" ) ) #Obrim l'arxiu amb les Im ID'
    classi = open(results_dir + "classificacio.txt", "w") #Creem l'arxiu per guardar els resultats
    
    descrip = pickle.load(open(dir_descriptores+features_file, "r")) #Obrim l'arxiu per llegir les im ID's


    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    #For per asignar un edifici random despres de cada id després de una tabulació: 29796-14601-23662	catedral
    for keyval, valuesval in ImID.items():
        classi.write(keyval + "\t" + vector_etiquetes[np.random.randint(0, 13)]+ "\n") #Asignem una classificació aleatoria depenent de la posició (random) del vector d'etiquetes
        
    classi.close()
   
#definim un vector amb totes les posibles etiquetes:

vector_etiquetes = ["mnactec", "mercat_independencia", "ajuntament", "societat_general", "estacio_nord", "dona_treballadora", "escola_enginyeria", "catedral", "teatre_principal", "farmacia_albinyana", "masia_freixa", "castell_cartoixa", "desconegut"]


classify ("descriptor_train.p", dir_archivos_txt, vector_etiquetes)
