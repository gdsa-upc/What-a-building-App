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

def plot_confusion_matrix(cm, true_labels,normalize = False,title='Confusion matrix', cmap=plt.cm.Blues):
    
    # Normalize matrix rows to sum 1
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(true_labels))
    plt.xticks(tick_marks, true_labels, rotation=90)
    plt.yticks(tick_marks, true_labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.show()

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
    gt_val_ID[j]=label[0]#agafem només el nom de la id
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
aa_val_aux=[]
aa_val_ID_aux=[]
for i in aa_val:
    id_image = gt_val_ID[j]
    k=0
    for l in aa_val_ID:
        if l==id_image:
            aa_val_aux.append(aa_val[k])
            aa_val_ID_aux.append(aa_val_ID[k])
        k=k+1     
    j=j+1

aa_val=aa_val_aux
aa_val_ID=aa_val_ID_aux

# accuracy test
#y_true = [gt_val]
#y_pred = [aa_val]

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

cm = confusion_matrix(gt_val,aa_val)
classes = np.unique(gt_val) #llista de les classes
plt.figure(1)
plot_confusion_matrix(cm,classes,normalize=True)
