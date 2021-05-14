
import os
import cv2

class FileManager:

    def countNumberOfFiles(self, path):
        listPath = os.listdir(path)
        print("Numero de archivos: " + str(len(listPath)))
        return len(listPath)

    def loadTrainData(self, path, numberOfFiles):
        trainImagesArray = numberOfFiles * [0]
        for i in range(numberOfFiles - 1):
            trainImagesArray[i] = cv2.imread(path + "\\" + ('%05d' % i) + ".ppm")
            print("Readed " + path + "\\" + ('%05d' % i) + ".ppm")
        return trainImagesArray

    def loadTestData(self, path, numberOfFiles):
        testImagesArray = numberOfFiles * [0]
        for i in range(numberOfFiles):
            testImagesArray[i] = cv2.imread(path + "\\" + ('%05d' % (400 + i)) + ".jpg")
            print("Readed " + path + "\\" + ('%05d' % (400 + i)) + ".jpg")
        return testImagesArray