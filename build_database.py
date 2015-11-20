# -*- coding: utf-8 -*-
import cv2 #Llibreria: http://opencv.org/
import numpy as np # Llirebria: http://www.numpy.org/
import matplotlib.pyplot as plt
from scipy.misc import imsave
from scipy import misc
from glob import glob


# Lee el fichero (annotation.txt )donde aparecen el nombre de la clase con el nombre del archivo
# y lo almacena en dos variables. array_id (numero) y array_clase(catedral, etc..). Despues lo escribe en un fichero .txt

#450 imagenes en la carpeta train
#180 imagenes en la carpeta val

num_picture_train=450+1;
num_picture_val=180+1;

i=0 #variable de iteración para guardar cada palabra
array_id = ["" for x in range(num_picture_val)] # Vector que almacena las id's aleatorias
array_clase = ["" for x in range(num_picture_val)] #Vector que almacena el nombre de la case (catedral...etc.)
fichero_final=open("Text_ID_picture.txt","w") # abre un fichero
with open('./TerrassaBuildings900/val/annotation.txt','r') as fichero:
    ##linia= fichero.readlines()
    for linea in fichero: # mientras haya lineas..
        words = linea.split() # lee las palabras de cada linia haciendo un split()
                                # la variable words tiene las posiciones que palabras haya en la linia, en este caso dos
        array_id[i]=words[0] # almacena la id aleatoria
        array_clase[i]=words[1] # almaena el nombre de la clase
        fichero_final.write(words[0]+"\n") #escribe la id en el fichero por cada fichero
        i=i+1 # incrementa la iteración para que el vector tenga las posiciones que debe tener(180)


fichero_final.close() # cierra el fichero_final


#Pruebas para comprobar que la id[1] corresponde al nombre clase[1]
#Las posiciones 0 quedan utilizadas para almacenar el nombre de los campos ImageID y ClassID

print("Image ID: " + array_id[1])
print("Class ID: " + array_clase[1])

print(words) # muestra la última posicion la 180 del vector, que es la ultima que ha leido

print("Image ID: " + array_id[180])
print("Class ID: " + array_clase[180])
