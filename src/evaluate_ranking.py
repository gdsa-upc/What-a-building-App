# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os

#    Para utilizar el path se debe utilizar una estructura asi:

#    TerrassaBuildings900               -> Aquí deben estar las imágenes
#    src                                -> Aquí deben estar los archivos .py
#    txt                                -> Aquí guardaremos los ficheros txt generados
#    rank                               -> Lista de ficheros. Cada uno contiene un ranking de las imagenes de training 


path_imagenes_train = "../TerrassaBuildings900/train/images/"
path_imagenes_val = "../TerrassaBuildings900/val/images/"
dir_archivos_txt = "../txt/"
dir_descriptores = "../descriptores/"
dir_rank= "../rank/"
dir_princ="../"


def evaluate_rank(dir_rank):
    nfiles = os.listdir(dir_rank) 
    ground_truth_val = open("../TerrassaBuildings900/val/annotation.txt", "r")
    ground_truth_train = open("../TerrassaBuildings900/train/annotation.txt","r")
    truth = {} #inicialitzem una taula on l'index es la id de la imatge i conté la seva categoria
    AP = {}
    next(ground_truth_val)#eliminem la primera linia de l'arxiu ja que no ens interessa
    next(ground_truth_train)#eliminem la primera linia de l'arxiu ja que no ens interessa
    for line in ground_truth_val:
        id_foto = line.index("\t")
        final = line.index("\n")
        truth[line[0:id_foto]] = line[id_foto+1:final] #guardem la categoria de cada imatge a un vector
    for line in ground_truth_train:
        id_foto = line.index("\t")
        final = line.index("\n")
        truth[line[0:id_foto]] = line[id_foto+1:final] #guardem la categoria de cada imatge a un vector
    MAP = 0
    APC = 0

    for file in nfiles:
        ranking = open(dir_rank+"/"+file,"r")#obrim l'arxiu rank d'una imatge de cerca
        filename = file[5:file.index(".txt")] #Amb el numero 5, no agafem la paraula "rank_" 
        categoria = truth[filename] #assignem la categoria que té la imatge de cerca
        relevants = 0
        precision = 0
        AP[filename] = 0
        irrelevants = 0
        k = 0
        for line in ranking:
            final = line.index("\n")
            k += 1
            if truth[line[0:final]] == categoria: #si la id de la imatge coincideix amb la categoria sumem + 1 a relevants
                relevants += 1 
                precision = precision + float(relevants)/float(k) #calculem la precisio per cada k
            else:
                irrelevants += 1
        AP[filename] = float(precision)/float(relevants) #calculem la AP de cada imatge de cerca
        APC += AP[filename]  #calculem la AP acumulada de cada imatge de cerca

        ranking.close()
    MAP = APC/len(nfiles) #calcul del MAN
    return AP, MAP #retornem els valors de AP de cada imatge i de MAN


AP,MAP = evaluate_rank(dir_rank)

print AP

print "\nMAP: " + str(MAP) + "\n"
