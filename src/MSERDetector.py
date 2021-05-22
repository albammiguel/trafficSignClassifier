import cv2
from typeSignEnum import typeSignEnum
import numpy as np

class MSERDetector:

    def cropResizedImage(self, image, x1, x2, y1, y2):
        # cortamos el area de la imagen donde está la señal.
        cropp_image = image[y1:y2, x1:x2]

        # redimensionamos el area recortada.
        resized_image = cv2.resize(cropp_image, (25, 25), interpolation=cv2.INTER_AREA)

        return resized_image



    def getSignByTypeList(self, numberTrainFiles, imagesTrain, trainInfoImagesArray):

        prohibitionImagesList = []
        dangerImagesList = []
        stopImagesList = []

        for i in range(numberTrainFiles-1):
            image = imagesTrain[i]
            image_info = trainInfoImagesArray[i]
            list_signs = getattr(image_info, 'listSignInfo')
            cont = 0
            for sign in list_signs:
                print("Señal " + str(cont) + " de la imagen " + getattr(image_info, 'image'))
                print("-----------")
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
        img_hsv = cv2.cvtColor(avg_image, cv2.COLOR_BGR2HSV)

        #obtener mascara de los valores mas bajos de rojo.
        mask1 = cv2.inRange(img_hsv, (0, 50, 20), (5, 255, 255))
        # obtener mascara de los valores mas altos de rojo.
        mask2 = cv2.inRange(img_hsv, (175, 50, 20), (180, 255, 255))

        return (mask1 | mask2)
