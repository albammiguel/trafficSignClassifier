
class RegionDetectedInfo:
    image = None
    score = 0
    signInfo = None

    def __init__(self, image, signInfo):
        self.image = image
        self.signInfo = signInfo


