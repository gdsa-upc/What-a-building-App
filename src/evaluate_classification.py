# -*- coding: utf-8 -*-
import cv2
import pickle #Ejemplos de serialización: https://docs.python.org/2/library/pickle.html
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

#    Para utilizar el path se debe utilizar una estructura asi:

#    TerrassaBuildings900               -> Aquí deben estar las imágenes
#    src                                -> Aquí deben estar los archivos .py
#    txt                                -> Aquí guardaremos los ficheros txt generados


path_imagenes_train = "../TerrassaBuildings900/train/images/"
path_imagenes_val = "../TerrassaBuildings900/val/images/"
dir_archivos_txt = "../txt/"
dir_descriptores = "../descriptores/"


f = open("../TerrassaBuildings900/val/annotation.txt", "r") #Obrim l'arxiu per llegir el validation de ground truth
gt_val = f.readlines() #creem una llista amb cada un dels elements del .txt
f.close()

f = open("../TerrassaBuildings900/val/annotation.txt", "r") #Obrim l'arxiu per llegir el validation de ground truth
gt_val_ID = f.readlines()
f.close()

j=0
gt_val.pop(0) #eliminem la primera linia (ImageID ClassID)
gt_val_ID.pop(0)

for i in gt_val:
    label = i.replace('\n','') #eliminem els \n
    label=label.split('\t') #separem les linies en dos a partir del \t
    gt_val_ID[j]=label[0]
    label=label[1] #agafem només el nom de la classe
    gt_val[j]=label #guardem al vector només el nom de l'etiqueta
    j=j+1

f = open("../txt/classification.txt", "r") #Obrim l'arxiu val de automatic annotation
aa_val = f.readlines()
f.close()

f = open("../txt/classification.txt", "r") 
aa_val_ID = f.readlines()
f.close()

j=0

for i in aa_val:
    label = i.replace('\n','') 
    label=label.split('\t') 
    aa_val_ID[j]=label[0]
    label=label[1] 
    aa_val[j]=label 
    j=j+1
     
j=0
prova=[]
prova_ID=[]
for i in aa_val:
    w = gt_val_ID[j]
    k=0
    for l in aa_val_ID:
        if l==w:
            prova.append(aa_val[k])
            prova_ID.append(aa_val_ID[k])
        k=k+1     
    j=j+1

aa_val=prova
aa_val_ID=prova_ID

# accuracy test
#y_true = [gt_train]
#y_pred = [aa_train]

print("Accuracy")
print( accuracy_score(gt_val, aa_val))
print("\n")

#Precision test 
#The precision is the ratio tp / (tp + fp)
print("Precision general")
print(precision_score(gt_val, aa_val, average='macro')) #ho calcula per tots
print("\n")

print("Precision per cada classe")
print(precision_score(gt_val, aa_val, average=None)) #ho calcula per cada classe
print("\n")


#recall test
# recall is the ratio tp / (tp + fn) 

print("Recall general")
print(recall_score(gt_val, aa_val, average='macro')) #ho calcula per tots
print("\n")

print("Recall per cada classe")
print(recall_score(gt_val, aa_val, average=None)) #ho calcula per cada classe
print("\n")

#F1 score test
# F1 = 2 * (precision * recall) / (precision + recall)
print("F1-score en general")
print(f1_score(gt_val, aa_val, average='macro')) #ho calcula per tots
print("\n")

print("F1-score per cada classe")
print(f1_score(gt_val, aa_val, average=None)) #ho calcula per cada classe
print("\n")

#confusion matrix test
print("Matriu confusió")
print(confusion_matrix(gt_val, aa_val))
