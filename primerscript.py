<<<<<<< Updated upstream
#Iportar la llibreria de OpenCV2
=======
#Leer las librerias
>>>>>>> Stashed changes
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

<<<<<<< Updated upstream
#Importem l'imatge i la convertim a nivells de gris
img = cv2.imread('/home/cmanso/Documents/clase/GDSA/projecte/jl.jpg')
=======
#elegir un patch
patch="/User/Sergi/Desktop/GDSA101.1/101.1/"

#Leer los datos 
img = cv2.imread(patch+'IMG_7057.jpg')
#plt.imshow(img),plt.show()
>>>>>>> Stashed changes
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Creem un objecte de sift i el fem servir per detectar els punts d'interes
sift = cv2.SIFT()
kp = sift.detect(gray,None)

#Dibuixem els punts d'interes a l'imatge
img=cv2.drawKeypoints(gray,kp)

#Desem l'imatge
cv2.imwrite('/home/cmanso/Documents/clase/GDSA/projecte/sift_keypoints.jpg',img)
