#Iportar la llibreria de OpenCV2
import cv2
import numpy as np

#Importem l'imatge i la convertim a nivells de gris
img = cv2.imread('/home/cmanso/Documents/clase/GDSA/projecte/jl.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Creem un objecte de sift i el fem servir per detectar els punts d'interes
sift = cv2.SIFT()
kp = sift.detect(gray,None)

#Dibuixem els punts d'interes a l'imatge
img=cv2.drawKeypoints(gray,kp)

#Desem l'imatge
cv2.imwrite('/home/cmanso/Documents/clase/GDSA/projecte/sift_keypoints.jpg',img)
