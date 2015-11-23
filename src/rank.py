# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np

# Calcular distancia: POR IMPLEMENTAR
def distance(x, y): 
    return np.random.randint(0, 451)

def rank(features_file, results_dir, features_train, annotation_path):
    features_val = pickle.load( open("../descriptores/"+features_file, "r" ) )
    features_train = pickle.load( open("../descriptores/"+features_train, "r" ) )
    

    # Creamos el directorio si no existe
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Por cada imagen en el set de validación, creamos un ranking de las imagenes de training y lo escribimos en fichero
    vector= []
    for k_val, v_val in features_val.items():
        annotations= open(annotation_path, "r")
        for line in annotations:
            rec= line.split("\t")
            if rec[0]==k_val and rec[1]!="desconegut\n":
                rank= open(results_dir + "/rank_" + k_val +'.txt', 'w')
                for k_train, v_train in features_train.items():
                    vector.insert( distance(v_train[0], v_val[0]), k_train) 
                for item in vector:
                    rank.write("%s\n" % item)
                vector=[]
                rank.close()

# Ejecutamos
rank("descriptor_val.p", "../rank/", "descriptor_train.p", "../TerrassaBuildings900/val/annotation.txt")
