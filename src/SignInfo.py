
class SignInfo:
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    tipo = 0

    def __init__(self, x1, y1, x2, y2, tipo):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.tipo = tipo

    def printSign(self):
        print(" x1: " + str(self.x1))
        print(" y1: " + str(self.y1))
        print(" x2: " + str(self.x2))
        print(" y2: " + str(self.y2))
        print(" tipo: "+ str(self.tipo))

