from SignInfo import SignInfo


class RegionDetectedInfo(SignInfo):
    image = None
    score = 0

    def __init__(self, image, x1, x2, y1, y2, tipo):
        SignInfo.__init__(self, x1, x2, y1, y2, tipo)
        self.image = image

    def printRegionDetected(self):
        print("score: " + str(self.score))
        SignInfo.printSign(self)


