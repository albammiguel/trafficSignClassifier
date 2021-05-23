from ImageInfo import ImageInfo
from SignInfo  import SignInfo
import os
import cv2

class FileManager:

    def countNumberOfFiles(self, path):
        listPath = os.listdir(path)
        print("Numero de archivos: " + str(len(listPath)))
        return len(listPath)

    def loadTrainData(self, path, numberOfFiles):
        trainImagesArray = numberOfFiles * [0]
        trainInfoImagesArray = numberOfFiles * [0]
        trainFile = open(path + "\\" + "gt.txt", "r").readlines()
        for i in range(numberOfFiles - 1):
            trainImagesArray[i] = cv2.imread(path + "\\" + ('%05d' % i) + ".ppm")
            print("Readed " + path + "\\" + ('%05d' % i) + ".ppm")
            trainInfoImagesArray[i] = self.loadInfoSigns(trainFile, ('%05d' % i) + ".ppm")
        return trainImagesArray, trainInfoImagesArray

    def loadTestData(self, path, numberOfFiles):
        testImagesArray = numberOfFiles * [0]
        for i in range(numberOfFiles):
            testImagesArray[i] = cv2.imread(path + "\\" + ('%05d' % (400 + i)) + ".jpg")
            print("Readed " + path + "\\" + ('%05d' % (400 + i)) + ".jpg")
        return testImagesArray

    def loadInfoSigns(self, trainFile, name):
        listSigns = []
        cont = 0
        for i in range(len(trainFile)):
            infoLine = trainFile[i].split(";")
            if(infoLine[0] == name):
                sign = SignInfo(infoLine[1],infoLine[2],infoLine[3],infoLine[4],infoLine[5])
                listSigns.append(sign)
                ++cont

        image = ImageInfo(name,cont,listSigns)
        return image




