from ImageInfo import ImageInfo
from SignInfo  import SignInfo
import os
import cv2

class FileManager:

    #método que cuenta el número de archivos que hay en un directorio.
    def countNumberOfFiles(self, path):
        listPath = os.listdir(path)
        print("Numero de archivos: " + str(len(listPath)))
        return len(listPath)

    #método que carga los archivos de train y guarda su información en los respectivos objetos o estructuras de datos necesarias.
    def loadTrainData(self, path, numberOfFiles):
        trainImagesArray = numberOfFiles * [0]
        trainInfoImagesArray = numberOfFiles * [0]
        trainFile = open(path + "\\" + "gt.txt", "r").readlines()
        for i in range(numberOfFiles - 1):
            trainImagesArray[i] = cv2.imread(path + "\\" + ('%05d' % i) + ".ppm")
            print("Readed " + path + "\\" + ('%05d' % i) + ".ppm")
            trainInfoImagesArray[i] = self.loadInfoSigns(trainFile, ('%05d' % i) + ".ppm")
        return trainImagesArray, trainInfoImagesArray

    # método que carga los archivos de test y guarda su información en los respectivos objetos o estructuras de datos necesarias.
    def loadTestData(self, path, numberOfFiles):
        testImagesArray = numberOfFiles * [0]
        for i in range(numberOfFiles):
            testImagesArray[i] = cv2.imread(path + "\\" + ('%05d' % (400 + i)) + ".jpg")
            print("Readed " + path + "\\" + ('%05d' % (400 + i)) + ".jpg")
        return testImagesArray

    #método que extrae la información de las señales del txt de las imagenes de train
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

    #método que genera los ficheros resultados.
    def generateResultFile(self, path, textFile):
        if(os.path.isfile(path)):
            f = open(path, "a")
            f.write(textFile)
            f.close()
        else:
            f = open(path, "w")
            f.write(textFile)
            f.close()

    #método que genera el directorio con las imágenes.
    def saveImageInDirectory(self, path, image, nameImage):
        if(os.path.isdir(path)):
            cv2.imwrite(os.path.join(path, nameImage), image)
        else:
            os.mkdir(path)
            cv2.imwrite(os.path.join(path, nameImage), image)





