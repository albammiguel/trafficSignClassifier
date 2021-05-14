import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

def loadData(path, numberOfFiles):
    imagesArray = numberOfFiles * [0]
    for i in range(numberOfFiles):
        imagesArray[i] = cv2.imread(path + numberOfFiles + ".ppm")
    return imagesArray



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="", help='Detector string name')
    parser.add_argument(
        '--train_path', default="", help='Select the training data dir')
    parser.add_argument(
        '--test_path', default="", help='Select the testing data dir')

    args = parser.parse_args()

    image = cv2.imread('C://Users//albam//Desktop//URJC//SEGUNDO_CUATRIMESTRE//VISION_ARTIFICIAL//PRACITCA_1//train//train//00398.ppm')
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(image_grey, 180, 200, cv2.THRESH_BINARY)
    cv2.imshow('umbralizado', binary)
    mser = cv2.MSER_create(_delta=50, _max_variation=0.5, _max_area=10000)
    polygons = mser.detectRegions(binary)

    for p in polygons[0]:
        x,y,w,h = cv2.boundingRect(p)
        if(w > 1 and h > 1):
            cv2.rectangle(image,(x,y),(x+w,y+h),(155,155,0),1)
    cv2.drawContours(binary,polygons[0],-1,(0,0,255),3)

    cv2.imshow('2', image)

    cv2.waitKey()








    # Load training data
    # Create the detector
    # Load testing data
    # Evaluate sign detections





