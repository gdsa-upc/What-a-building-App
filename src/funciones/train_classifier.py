#LLIBRERIES

import numpy as np
import os
import pickle
from sklearn.svm import SVC

#DEFINIM EL PATH en la variable ruta
ruta = os.path.dirname(os.path.abspath(__file__))

def train_classifier("""features_train""","""dir_model""","""val_or_train"""):

    #agafem fitxer d'anotacions
    nfiles_t = open("""insertar ruta""") 
    
    #agafem fitxer de caracteristiques
    features_train =open(ruta+"/files/outfile_"+val_or_train+".txt","r") 
    
    #agafem fitxer de sortida
    dir_model = open(ruta+"/files/trained_model.p","w") 

    #guardem el fitxer d'annotacions en una variable en format diccionari
    features_train = pickle.load(nfiles_t)
    
    #creem el diccionari (buit)
    train_mod = dict() 
    
    #definim variable de contador
    i = 0
    
    for kval in nfiles_t:
      dir_model.write(kval+"\t")
      for vector in features_train:
        if (i<=99)
        dir_model.write(vector+"\t")
        i +=1
      i=0  
      dir_model.write("\n")
    dir_model.close()
    train_mod.close()
    features_train.close()

#CRIDA A LA FUNCIÓ
train_classifier(ruta+"/files/outfile_train.txt",ruta+"/files/trained_model.p","train");
                        
