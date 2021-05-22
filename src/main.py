import argparse
from FileManager import FileManager
from MSERDetector import MSERDetector
import cv2
import numpy as np

# import matplotlib.pyplot as plt



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

    # Load training data
    fileManager = FileManager()
    numberTrainFiles = fileManager.countNumberOfFiles(args.train_path)
    imagesTrain, trainInfoImagesArray = fileManager.loadTrainData(args.train_path, numberTrainFiles)

    # Create the detector
    if(args.detector == "mser"):
        mserDetector = MSERDetector()

        prohibitionImagesList, dangerImagesList, stopImagesList = mserDetector.getSignByTypeList(numberTrainFiles, imagesTrain, trainInfoImagesArray)
        avg_prohibition, avg_danger, avg_stop = mserDetector.calculateMean(prohibitionImagesList), mserDetector.calculateMean(dangerImagesList), mserDetector.calculateMean(stopImagesList)
        mask_prohibition, mask_danger, mask_stop = mserDetector.createMask(avg_prohibition), mserDetector.createMask(avg_danger), mserDetector.createMask(avg_stop)
        cv2.imshow('mascara prohibition', mask_prohibition)
        cv2.waitKey()
        cv2.imshow('mascara danger', mask_danger)
        cv2.waitKey()
        cv2.imshow('mascara stop', mask_stop)
        cv2.waitKey()

    # Load testing data
    numberTestFiles = fileManager.countNumberOfFiles(args.test_path)
    imagesTest = fileManager.loadTestData(args.test_path, numberTestFiles)

    # Evaluate sign detections

    for i in range(numberTestFiles):
        sectionsList = []
        image = imagesTest[i+41]
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_umbralize = cv2.adaptiveThreshold(image_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
        #cv2.imshow('umbralizado', image_umbralize)
        mser = cv2.MSER_create(_delta=60, _max_variation=1, _max_area=1250, _min_area=90)
        polygons = mser.detectRegions(image_umbralize)
        for p in polygons[0]:
            x, y, w, h = cv2.boundingRect(p)
            if (0.8 <= w / h <= 1.5):
                print("Imagen "+str(i))
                #hacemos la región detectada mayor.
                w = w+ int(w/2)
                x = x-int(w/5)
                h = h+ int(h/2)
                y = y-int(h/5)

                crop_image = mserDetector.cropResizedImage(image, x, (x+w), y, (y+h))
                mask_image = mserDetector.createMask(crop_image)

                if(len(crop_image) != 0):
                    cv2.imshow('imagen recortada', crop_image)
                    cv2.waitKey()
                    cv2.imshow('mascara', mask_image)
                    cv2.waitKey()

                    sectionsList.append(mask_image)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)


        """cv2.imshow('2', image)
        cv2.waitKey()"""

