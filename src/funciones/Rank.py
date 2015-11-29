# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np

# Calculalem la 'euclidean' distance:
def euclidean(x, y): 
    dist = sklearn.metrics.pairwise.pairwise_distances(x,y)
    return dist
    
#capçelera rank
def rank(features_file, features_val, features_train, annotation_path):
    
    #carreguem els paths
    features_val = pickle.load( open("../descriptores/"+features_file, "r" ) )
    features_train = pickle.load( open("../descriptores/"+features_train, "r" ) )
    
    #En el cas que no existeixi el directori el creem
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    #Per cada Bow, comparem amb la 'euclidean' distance quin s'apropa més a la imatge de training
    v = []
    for kval, vval in features_val.items():
        annotations= open(annotation_path, "r")
        for line in annotations:
            rec= line.split("\t")
            if rec[0]==kval and rec[1]!="desconegut\n":
                rank= open(results_dir + "/rank_" + kval +'.txt', 'w')
                for ktrain, vtrain in features_train.items():
                    vector.insert(euclidean(vtrain[0], vval[0]), ktrain) 
                for item in vector:
                    rank.write("%s\n" % item)
                v=[]
                rank.close()

#Executem la funcio Rank amb els paths correctes.
rank("descriptor_val.p", "../rank/", "descriptor_train.p", "../TerrassaBuildings900/val/annotation.txt")
