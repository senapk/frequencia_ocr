import cv2
import numpy as np


img = cv2.imread("folha.jpg")

# Segurança: verifica se carregou
if img is None:
    raise ValueError("Imagem não carregada.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# aumentar contraste
gray = cv2.equalizeHist(gray)

# remover ruído
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# fazer contornos
# gray = cv2.Canny(gray, 50, 150, apertureSize=3)





cv2.imwrite("saida.jpg", gray)
