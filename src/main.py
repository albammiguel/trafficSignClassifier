import argparse
from FileManager import FileManager
from MSERDetector import MSERDetector
import cv2
import numpy as np
from scipy import signal
from RegionDetectedInfo import RegionDetectedInfo
from SignInfo import SignInfo

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
        mserDetector.executeDetector(numberTrainFiles, imagesTrain, trainInfoImagesArray)

    # Load testing data
    numberTestFiles = fileManager.countNumberOfFiles(args.test_path)
    imagesTest = fileManager.loadTestData(args.test_path, numberTestFiles)

    # Evaluate sign detections
    if (args.detector == "mser"):
        mserDetector.evaluateSignDetections(numberTestFiles, imagesTest, fileManager)




    """cv2.imshow('2', image)
        cv2.waitKey()"""

