import argparse
from FileManager import FileManager
from MSERDetector import MSERDetector
import cv2
import numpy as np

# import matplotlib.pyplot as plt
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

        prohibitionImagesList, dangerImagesList, stopImagesList = mserDetector.getSignByTypeList(numberTrainFiles, imagesTrain, trainInfoImagesArray)
        avg_prohibition, avg_danger, avg_stop = mserDetector.calculateMean(prohibitionImagesList), mserDetector.calculateMean(dangerImagesList), mserDetector.calculateMean(stopImagesList)
        mask_prohibition, mask_danger, mask_stop = mserDetector.createMask(avg_prohibition), mserDetector.createMask(avg_danger), mserDetector.createMask(avg_stop)

    # Load testing data
    numberTestFiles = fileManager.countNumberOfFiles(args.test_path)
    imagesTest = fileManager.loadTestData(args.test_path, numberTestFiles)

    # Evaluate sign detections


    for i in range(numberTestFiles):
        detectedRegionsList = []
        image = imagesTest[196]
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_umbralize = cv2.adaptiveThreshold(image_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
        mser = cv2.MSER_create(_delta=60, _max_variation=1, _max_area=1250, _min_area=90)
        polygons = mser.detectRegions(image_umbralize)
        for p in polygons[0]:
            x, y, w, h = cv2.boundingRect(p)
            if (0.8 <= w / h <= 1.5):

                #hacemos la región detectada mayor.
                w = w + int(w/2)
                h = h + int(h/2)

                x1 = x-int(w/5)
                y1 = y-int(h/5)

                x2 = x1+w
                y2 = y1+h

                if(x1 >= 0 and y1>=0):
                    crop_image = mserDetector.cropResizedImage(image, x1, x2, y1, y2)

                mask_image = mserDetector.createMask(crop_image)
                sign = SignInfo(x1, y1, x2, y2, -1)

                if(len(crop_image) != 0):
                    region = RegionDetectedInfo(mask_image, sign)
                    detectedRegionsList.append(region)

        white_mask = 255 * np.ones((25,25), dtype=np.uint8)

        for sectionImg in detectedRegionsList:
            corrProhibition = cv2.matchTemplate(RegionDetectedInfo.__getattribute__(sectionImg, 'image'), mask_prohibition, cv2.TM_CCORR)
            corrDanger = cv2.matchTemplate(RegionDetectedInfo.__getattribute__(sectionImg, 'image'), mask_danger, cv2.TM_CCORR)
            corrStop = cv2.matchTemplate(RegionDetectedInfo.__getattribute__(sectionImg, 'image'), mask_stop, cv2.TM_CCORR)
            corrWhite = cv2.matchTemplate(RegionDetectedInfo.__getattribute__(sectionImg, 'image'), white_mask, cv2.TM_CCORR)
            cv2.imshow("mascara", RegionDetectedInfo.__getattribute__(sectionImg, 'image'))
            cv2.waitKey()

            print("---------- Imagen " + str(i) + "---------------")
            print("Correlacion prohibicion: "+ str(corrProhibition))
            print("Correlacion peligro: " + str(corrDanger))
            print("Correlacion stop: " + str(corrStop))
            print("Correlacion blanca: " + str(corrWhite))
            print("---------------------------------------")


    """cv2.imshow('2', image)
        cv2.waitKey()"""

