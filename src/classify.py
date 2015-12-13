# -*- coding: utf-8 -*-

import pickle #Ejemplos de serialización: https://docs.python.org/2/library/pickle.html
import numpy as np
import os



def classify(bow_val_path, classifier_path, classification_path):
    bow_val = pickle.load( open(bow_val_path, "r" ))
    classifier= pickle.load( open(classifier_path, 'r') )
    classification = open(classification_path, "w") 
    print bow_val


    for im_id, im_bow in bow_val.iteritems():
        classification.write(str(im_id) + "\t" + str(classifier.predict(im_bow)[0]) + "\n"  )
    classification.close()
   

classify ("../txt/bow_val.p", "../txt/classifier.p", "../txt/classification.txt")
