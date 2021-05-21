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
            trainInfoImagesArray[i] = ImageInfo(('%05d' % i) + ".ppm")
            print("Readed " + path + "\\" + ('%05d' % i) + ".ppm")
            self.loadInfoSigns(trainFile, trainInfoImagesArray[i])
        return trainImagesArray, trainInfoImagesArray

    def loadTestData(self, path, numberOfFiles):
        testImagesArray = numberOfFiles * [0]
        for i in range(numberOfFiles):
            testImagesArray[i] = cv2.imread(path + "\\" + ('%05d' % (400 + i)) + ".jpg")
            print("Readed " + path + "\\" + ('%05d' % (400 + i)) + ".jpg")
        return testImagesArray

    def loadInfoSigns(self, trainFile, trainImage):

        listSigns = []
        cont = 0
        for i in range(len(trainFile)):
            infoLine = trainFile[i].split(";")
            if(infoLine[0] == getattr(trainImage,'image') ):
                print(getattr(trainImage,'image'))
                sign = SignInfo(infoLine[1],infoLine[2],infoLine[3],infoLine[4],infoLine[5])
                sign.printSign()
                listSigns.append(sign)
                ++cont

        if (cont != 0):
            setattr(trainImage, cont, 'numberSigns')
            setattr(trainImage, listSigns, 'listSignInfo')





