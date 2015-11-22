# -*- coding: utf-8 -*-
import cv2
import pickle #Ejemplos de serialización: https://docs.python.org/2/library/pickle.html
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

#    Para utilizar el path se debe utilizar una estructura asi:

#    TerrassaBuildings900               -> Aquí deben estar las imágenes
#    src                                -> Aquí deben estar los archivos .py
#    txt                                -> Aquí guardaremos los ficheros txt generados


path_imagenes_train = "../TerrassaBuildings900/train/images/"
path_imagenes_val = "../TerrassaBuildings900/val/images/"
dir_archivos_txt = "../txt/"
dir_descriptores = "../descriptores/"

f = open("../TerrassaBuildings900/val/annotation.txt", "r") #Obrim l'arxiu per llegir el validation de ground truth
gt_val = f.readlines()
f.close()

g = open("../TerrassaBuildings900/train/annotation.txt", "r") #Obrim l'arxiu per llegir el training de ground truth
gt_train = g.readlines()
g.close()

h = open("../txt/classificacio.txt", "r") #Obrim l'arxiu per llegir el training de ground truth
classi = h.readlines()
h.close()

# accuracy test
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 1, 2, 0, 0, 1]
print("accuracy")
print( accuracy_score(y_true, y_pred))
print("\n")

# average precision test
y2_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])

print("average precision")
print(average_precision_score(y2_true, y_scores) )
print("\n")
#recall test
#el recall és el resultat de la divisió dels True Positive entre la suma dels 
#TP i els false negative, en aquest exemple TP = 4 i FN = 2

print("Recall")
recall_score(y_true, y_pred, average=None)
print(recall_score(y_true, y_pred, average='macro'))
print("\n")

#F1 score test
# F1 = 2 * (precision * recall) / (precision + recall)
print("F1-score")
print(f1_score(y_true, y_pred, average='macro'))

print("\n")

#confusion matrix test
print("Matriu confusió")
print(confusion_matrix(y_true, y_pred))

y4_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
y4_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
confusion_matrix(y4_true, y4_pred, labels=["ant", "bird", "cat"])
print("\n")
