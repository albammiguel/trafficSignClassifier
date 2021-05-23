
import cv2

from RegionDetectedInfo import RegionDetectedInfo
from typeSignEnum import typeSignEnum
import numpy as np

class MSERDetector:

    mask_prohibition = None
    mask_danger = None
    mask_stop = None

    def cropResizedImage(self, image, x1, x2, y1, y2):
        # cortamos el area de la imagen donde está la señal.
            cropp_image = image[y1:y2, x1:x2]

        # redimensionamos el area recortada.
            if (len(cropp_image) != 0):
                resized_image = cv2.resize(cropp_image, (25, 25), interpolation=cv2.INTER_AREA)
                return resized_image
            else:
                return []

    def getSignByTypeList(self, numberTrainFiles, imagesTrain, trainInfoImagesArray):

        prohibitionImagesList = []
        dangerImagesList = []
        stopImagesList = []

        for i in range(numberTrainFiles-1):
            image = imagesTrain[i]
            image_info = trainInfoImagesArray[i]
            list_signs = getattr(image_info, 'listSignInfo')
            for sign in list_signs:
                x1, y1 = int(getattr(sign, 'x1')), int(getattr(sign, 'y1'))
                x2, y2 = int(getattr(sign, 'x2')), int(getattr(sign, 'y2'))
                tipo = int(getattr(sign, 'tipo'))
                cv2.rectangle(image, (x1, y1), (x2, y2), (155, 155, 0), 1)

                resized_image = self.cropResizedImage(image, x1, x2, y1, y2)

                if(tipo in typeSignEnum.STOP.value):
                    stopImagesList.append(resized_image)

                if (tipo in typeSignEnum.DANGER.value):
                    dangerImagesList.append(resized_image)

                if (tipo in typeSignEnum.PROHIBITION.value):
                    prohibitionImagesList.append(resized_image)

            """cv2.imshow('imagen ' + str(i) + ':', image)
            cv2.waitKey()"""

        return prohibitionImagesList, dangerImagesList, stopImagesList

    def calculateMean(self, list):
        avg_img = np.mean(list, axis=0)
        return avg_img.astype(np.uint8)


    def createMask(self,avg_image):
        if (len(avg_image) != 0):
            img_hsv = cv2.cvtColor(avg_image, cv2.COLOR_BGR2HSV)

            #obtener mascara de los valores mas bajos de rojo.
            mask1 = cv2.inRange(img_hsv, (0, 50, 0), (5, 255, 255))
            # obtener mascara de los valores mas altos de rojo.
            mask2 = cv2.inRange(img_hsv, (175, 50, 0), (180, 255, 255))

            return (mask1 | mask2)
        else:
            return []

    def executeDetector(self, numberTrainFiles, imagesTrain, trainInfoImagesArray):
        prohibitionImagesList, dangerImagesList, stopImagesList = self.getSignByTypeList(numberTrainFiles, imagesTrain, trainInfoImagesArray)
        avg_prohibition, avg_danger, avg_stop = self.calculateMean(prohibitionImagesList), self.calculateMean(dangerImagesList), self.calculateMean(stopImagesList)
        self.mask_prohibition, self.mask_danger, self.mask_stop = self.createMask(avg_prohibition), self.createMask(avg_danger), self.createMask(avg_stop)



    def calculateCorrelationScore(self, mask_1, mask_2):
        correlation = 0
        for i in range(25):
            for j in range(25):
                correlation = correlation + mask_1[i, j] * mask_2[i, j]

        return correlation

    def evaluateSignDetections(self, numberTestFiles, imagesTest):

        for i in range(numberTestFiles):
            detectedRegionsList = []
            image = imagesTest[i]
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
                        crop_image = self.cropResizedImage(image, x1, x2, y1, y2)

                    mask_image = self.createMask(crop_image)

                    if(len(crop_image) != 0):
                        region = RegionDetectedInfo(mask_image, x1, y1, x2, y2, -1)
                        detectedRegionsList.append(region)

            self.detectSign(detectedRegionsList, image)



    def detectSign(self, detectedRegionsList, image):

        white_mask = 255 * np.ones((25,25), dtype=np.uint8)

        for sectionImg in detectedRegionsList:

            mask_image = RegionDetectedInfo.__getattribute__(sectionImg, 'image')

            corrProhibition = self.calculateCorrelationScore(mask_image, self.mask_prohibition)
            corrDanger = self.calculateCorrelationScore(mask_image, self.mask_danger)
            corrStop = self.calculateCorrelationScore(mask_image, self.mask_stop)
            corrWhite = self.calculateCorrelationScore(mask_image, white_mask)

            isWhite = False
            if(corrWhite >= 200):
                detectedRegionsList.remove(sectionImg)
                isWhite = True
                print("la imagen es blanca")
            elif (max(corrDanger, corrProhibition, corrStop) >= 40 and not(isWhite)):
                RegionDetectedInfo.__setattr__(sectionImg, 'tipo', 1)
                RegionDetectedInfo.__setattr__(sectionImg, 'score', max(corrDanger, corrProhibition, corrStop))
                print("----Señal detectada-----")
                RegionDetectedInfo.printRegionDetected(sectionImg)

                x1 = RegionDetectedInfo.__getattribute__(sectionImg, 'x1')
                x2 = RegionDetectedInfo.__getattribute__(sectionImg, 'x2')
                y1 = RegionDetectedInfo.__getattribute__(sectionImg, 'y1')
                y2 = RegionDetectedInfo.__getattribute__(sectionImg, 'y2')

                cv2.rectangle(image, (x1,y1), (x2, y2), (0, 0, 255), 1)

        cv2.imshow("imagen", image)
        cv2.waitKey()

