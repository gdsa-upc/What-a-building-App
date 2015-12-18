# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
from sklearn.metrics import pairwise_distances

def rank(train_bow_path, val_bow_path, results_dir, annotation_path):
    train_bow = pickle.load( open(train_bow_path, "r") )
    val_bow = pickle.load( open(val_bow_path, "r") )
    

    # Creamos el directorio si no existe
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    i=0
    annotations= open(annotation_path, "r")
    for val_id, val_key in val_bow.items():
        annotations.seek(0)
        rank= {}
        for line in annotations:
            rec= line.split("\t")
            print rec[0] + " = " + val_id + "\n"
            print rec[1]
            if rec[0]== val_id and rec[1]!= "desconegut\n":
                for train_id, train_key in train_bow.items():
                    rank[train_id]= pairwise_distances(val_key, train_key, metric='euclidean', n_jobs=1)
                rank_file= open(results_dir + "/rank_" + val_id + ".txt", 'w')
                for k, v in sorted(rank.items(), key=lambda (k,v ): (v,k) ):
                    rank_file.write(k)
                    rank_file.write("\n")
                rank_file.close()

# Ejecutamos
if __name__=="__main__":
    rank("../txt/bow_train.p", "../txt/bow_val.p", "../rank/", "../TerrassaBuildings900/val/annotation.txt")
