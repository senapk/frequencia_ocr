import cv2
import numpy as np
from color import Color
from numpy.typing import NDArray

class Filter:
    def __init__(self):
        pass

    # Filtro de binarização Otsu
    def otsu(self, img: NDArray[np.uint8], path: str) -> NDArray[np.uint8]:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # aumentar contraste
        # img = cv2.equalizeHist(img)

        # remover ruído
        # img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # binarização otsu (scanner de documentos)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # fazer contornos
        # gray = cv2.Canny(gray, 50, 150, apertureSize=3)

        cv2.imwrite(path, img)
        print("Imagem filtrada salva em", Color.green(path))
        return img