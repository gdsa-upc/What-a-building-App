import numpy as np
import os
from funciones import *
from scipy.cluster.vq import vq, kmeans, whiten
import os.path




#archivo image id es el nombre del archivo. En este caso hay dos ID_images_train.txt o ID_images_val.txt

def get_features(db_train_txt, db_val_txt, dir_train, dir_val):

    n_img_codebook= 999 #numero de imagenes que cogeras (para hacer pruebas), para el final con poner 999 vale
    n_img= 999
    n_centroides= 50 #debería ser 100 al final
    db_train = open(db_train_txt, 'r') #Abrir el archivo de con las ID's de las imangenes
    db_val = open(db_val_txt, 'r') #Abrir el archivo de con las ID's de las imangenes

    #if not os.path.exists(directorio_descriptores):
    #    os.makedirs(directorio_descriptores)
    vec_features=[]
    
    i=0
    if (os.path.isfile("../txt/codebook.p") == False):
        for line in db_train:
            if i>=n_img_codebook:
                break
            print "Codebook: " + str(i) + "\n"
            i+=1
            im_id = line[0:-1]
            ruta= "../TerrassaBuildings900/train/images/" + str(im_id)

            if os.path.isfile(ruta + ".jpg"):
                features= whiten(get_local_features(ruta + ".jpg"))
            else:
                features= whiten(get_local_features(ruta + ".JPG"))
    
            for feat in features:
                vec_features.append(feat)

        codebook= train_codebook(n_centroides, vec_features)
        pickle.dump(codebook, open("../txt/codebook.p", "wb" ) )
        print "Codebook creado\n"
    else:
        codebook= pickle.load(open( "../txt/codebook.p", "rb"))
        print "Codebook cogido\n"

    i=0
    dic_train={}
    db_train.seek(0)
    for line in db_train:
        if i>=n_img:
            break
        i+=1
        #print "dic train: "+ str(i)+ "\n"
        print "feature_train: " + str(i) + "\n"
        im_id= line[0:-1]
        ruta= "../TerrassaBuildings900/train/images/" + str(im_id)
        if os.path.isfile(ruta + ".jpg"):
            features= whiten(get_local_features(ruta + ".jpg"))
        else:
            features= whiten(get_local_features(ruta + ".JPG"))

        assignments= get_assignments(codebook, features)
        #print assignments
        bag= build_bow(assignments, 100)
        dic_train[im_id]= bag
    dic_val={}
    i=0
    for line in db_val:
        if i>=n_img:
            break
        #print "dic_val: "+ str(i)+ "\n"
        print "feature_train: " + str(i) + "\n"

        i+=1
        im_id= line[0:-1]
        ruta= "../TerrassaBuildings900/val/images/" + str(im_id)
        if os.path.isfile(ruta + ".jpg"):
            features= whiten(get_local_features(ruta + ".jpg"))
        else:
            features= whiten(get_local_features(ruta + ".JPG"))

        assignments= get_assignments(codebook, features)
        bag= build_bow(assignments, 100)
        #print bag
        #print len(bag)
        dic_val[im_id]= bag
 
    pickle.dump(dic_train, open("../txt/bow_train.p", "wb" ) )
    pickle.dump(dic_val, open("../txt/bow_val.p", "wb" ) )


get_features("../txt/ID_images_train.txt", "../txt/ID_images_val.txt", "../TerrassaBuildings900/train/images/", "../TerrassaBuildings900/val/images/")

