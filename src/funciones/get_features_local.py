# -*- coding: utf-8 -*-
import cv2

#Transformació d'imatges: http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html

def get_features(imatge):

   # http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html

    img = cv2.imread(imatge,0)

    #gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #res = cv2.resize(img,((0.5)*width, (0.5)*height), interpolation = cv2.INTER_CUBIC)
    small = cv2.resize(img, (0,0), fx=0.1, fy=0.1) #Hacemos un risize i reducimos el tamaño de los ejes a un 10%

    sift = cv2.SIFT()
   # orb = cv2.ORB()

    #detectAndCompute: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

    kp, des = sift.detectAndCompute(small,None) #kp= número de punts d'interes -- des= descriptores

    num_descript = len(des)
    key_points = len(des[4]) # tamañao de un vector del descriptor
    print("Se han creado: " + str(num_descript) + " descriptores para la imagen" + imatge + " con "  + str(key_points) + " Keypoints por descriptor.")

    return des

#El fastfeaturedeteccrion: http://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_feature_detectors.html#fastfeaturedetector
#fast = cv2.FastFeatureDetector()

#des.size
#sys.getsizeof(kp)

#imatge2 = "../../archivosPrueba/IMG_7057.jpg"
#des = get_features(imatge2)
