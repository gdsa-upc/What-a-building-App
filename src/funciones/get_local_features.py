# -*- coding: utf-8 -*-
import cv2
import matplotlib as plt

#Esta funcion devuelve los descriptores de una imagen en formato float32

#Transformació d'imatges: http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html

def get_features(imatge):

    #Understand a image: http://www.weheartcv.com/understanding-image/

    # Informacion de SIFT :http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html

    #path = "../../TerrassaBuildings900/train/images/12507-3011-22068.JPG"

    img = cv2.imread(imatge)

    #gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    small = cv2.resize(img, (500,500)) #escalamos la imagen a 500x500 para
                        # Pruebas res = cv2.resize(img,((0.5)*width, (0.5)*height), interpolation = cv2.INTER_CUBIC)
                        #small = cv2.resize(img, (0,0), fx=0.1, fy=0.1) #Hacemos un risize i reducimos el tamaño de los ejes a un 10%
    sift = cv2.SIFT()
                        # Otro posible método: orb = cv2.ORB() #otra forma de extraccions

                        #detectAndCompute: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    kp, des = cv2.SIFT(99).detectAndCompute(small,None) #kp= número de punts d'interes -- des= descriptores
                        #Se limita 100 descriptores
                        #kp, des = orb.detectAndCompute(small,None) #kp= número de punts d'interes -- des= descriptores


    num_descript = len(des)
    key_points = len(des[4]) # tamañao de un vector del descriptor
    print("Se han creado: " + str(num_descript) + " descriptores para la imagen" + imatge + " con "  + str(key_points) + " Keypoints por descriptor.")

    return des

#if __name__ == "__main__":

#Pruebas:

#El fastfeaturedeteccrion: http://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_feature_detectors.html#fastfeaturedetector
#fast = cv2.FastFeatureDetector()

#des.size
#sys.getsizeof(kp)

#imatge2 = "../TerrassaBuildings900/train/images/24921-30622-26673.JPG"
#des = get_local_features(imatge2)

#print(len(des))
#print(len(des[1]))
