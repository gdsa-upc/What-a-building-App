# -*- coding: utf-8 -*-
import cv2
import pickle #Ejemplos de serialización: https://docs.python.org/2/library/pickle.html
import numpy as np
import os
from funciones.bag_of_words import bag_of_words
from funciones.get_local_features import get_local_features
from funciones.train_codebook import train_codebook
from funciones.get_assignments import get_assignments
from scipy.cluster.vq import vq, kmeans, whiten
import os.path




#archivo image id es el nombre del archivo. En este caso hay dos ID_images_train.txt o ID_images_val.txt

def get_features(db_train_txt, db_val_txt, dir_train, dir_val):

    n_rep= 5 #numero de imagenes que cogeras (para hacer pruebas)
    db_train = open(db_train_txt, 'r') #Abrir el archivo de con las ID's de las imangenes
    db_val = open(db_val_txt, 'r') #Abrir el archivo de con las ID's de las imangenes

    #if not os.path.exists(directorio_descriptores):
    #    os.makedirs(directorio_descriptores)

    vec_features=[]
    i=0
    for line in db_train:
        if i>=n_rep:
            break
        i+=1
        im_id = line[0:-1]
        ruta= "../TerrassaBuildings900/train/images/" + str(im_id)

        if os.path.isfile(ruta + ".jpg"):
            features= whiten(get_local_features(ruta + ".jpg"))
        else:
            features= whiten(get_local_features(ruta + ".JPG"))

        for feat in features:
            vec_features.append(feat)

    codebook= train_codebook(100, vec_features)

    dic_train={}
    i=0
    db_train.seek(0)
    for line in db_train:
        if i>=n_rep:
            break
        i+=1
        im_id= line[0:-1]
        ruta= "../TerrassaBuildings900/train/images/" + str(im_id)
        if os.path.isfile(ruta + ".jpg"):
            features= whiten(get_local_features(ruta + ".jpg"))
        else:
            features= whiten(get_local_features(ruta + ".JPG"))

        assignments= get_assignments(codebook, features)
        bag= bag_of_words(assignments)
        dic_train[im_id]= bag

    dic_val={}
    i=0
    for line in db_val:
        if i>=n_rep:
            break
        i+=1
        im_id= line[0:-1]
        ruta= "../TerrassaBuildings900/val/images/" + str(im_id)
        if os.path.isfile(ruta + ".jpg"):
            features= whiten(get_local_features(ruta + ".jpg"))
        else:
            features= whiten(get_local_features(ruta + ".JPG"))

        assignments= get_assignments(codebook, features)
        bag= bag_of_words(assignments)
        print bag
        print len(bag)
        dic_val[im_id]= bag
 
    pickle.dump(dic_train, open("../txt/bow_train.p", "wb" ) )
    pickle.dump(dic_val, open("../txt/bow_val.p", "wb" ) )


get_features("../txt/ID_images_train.txt", "../txt/ID_images_val.txt", "../TerrassaBuildings900/train/images/", "../TerrassaBuildings900/val/images/")

