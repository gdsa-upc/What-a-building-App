# -*- coding: utf-8 -*-
import cv2
import matplotlib as plt


# Devuelve los descriptores de una imagen en un array de float32
def get_local_features(imatge):

    # Otro posible método: orb = cv2.ORB()
    img = cv2.imread(imatge)


    small = cv2.resize(img, (500, 500)) #escalamos la imagen a 500x500
    
    # Pruebas res = cv2.resize(img,((0.5)*width, (0.5)*height), interpolation = cv2.INTER_CUBIC)
    #small = cv2.resize(img, (0,0), fx=0.1, fy=0.1) #Hacemos un risize i reducimos el tamaño de los ejes a un 10%


    #detectAndCompute: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    kp, des = cv2.SIFT().detectAndCompute(small, None) #kp= número de punts d'interes -- des= descriptores
    #kp, des = orb.detectAndCompute(small,None) #kp= número de punts d'interes -- des= descriptores
    print("Se han creado: " + str(len(des)) + " descriptores para la imagen " + "\"" + imatge + "\"" + " con "  + str(len(des[4])) + " Keypoints por descriptor.")
    return des

if __name__ == "__main__":
    image= "../TerrassaBuildings900/train/images/51-16676-22265.jpg"
    features= get_local_features(image)
    num_descript = len(features)
    key_points = len(features[4]) # tamaño de un vector del descriptor

   # print("Se han creado: " + str(num_descript) + " descriptores para la imagen " + "\"" + image + "\"" + " con "  + len(key_points) + " Keypoints por descriptor.")


