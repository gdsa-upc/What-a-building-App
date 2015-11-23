# -*- coding: utf-8 -*-
import cv2
import pickle #Ejemplos de serialización: https://docs.python.org/2/library/pickle.html
import numpy as np
import os
from sklearn.metrics import label_ranking_average_precision_score

from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

#    Para utilizar el path se debe utilizar una estructura asi:

#    TerrassaBuildings900               -> Aquí deben estar las imágenes
#    src                                -> Aquí deben estar los archivos .py
#    txt                                -> Aquí guardaremos los ficheros txt generados
#    rank                               -> Lista de ficheros. Cada uno contiene un ranking de las imagenes de training 


path_imagenes_train = "../TerrassaBuildings900/train/images/"
path_imagenes_val = "../TerrassaBuildings900/val/images/"
dir_archivos_txt = "../txt/"
dir_descriptores = "../descriptores/"
dir_rank= "../rank"
dir_princ="../"




def evaluate_ranking(directorio_lista, trainorval):    #definimos funcion con entradas directorio con la lista, y el tipo si es training or validation
    features_val = pickle.load( open("../rank"+kval,".txt", "r" ) )
    features_train = pickle.load( open("../descriptores/"+features_train, "r" ) )
    
    f= open("../TerrassaBuildings900/val/annotation.txt", "r") #Obrim l'arxiu per llegir el validation de ground truth
    val_annotation = f.readlines()
    f.close()
    
    g = open("../TerrassaBuildings900/train/annotation.txt", "r") #Obrim l'arxiu per llegir el training de ground truth
    train_annotation = g.readlines()
    g.close()
        
   
    for line in directorio_lista:      #per cada linia del directori rank...
       
    if trainorval=="val":
    AP = label_ranking_average_precision_score(directorio_lista, val_annotation)     # calcular la average precision en el cas que sigui val_annotation
    
    else if trainorval=="train":
    AP = label_ranking_average_precision_score(directorio_lista, train_annotation)  #calcular la average precision en cas que sigui train_annotation
    
    

#y_true = np.array([[1, 0, 0], [0, 0, 1]])
#y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
#label_ranking_average_precision_score(y_true, y_score)         
#0.416...




