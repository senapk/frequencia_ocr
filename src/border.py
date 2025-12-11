import numpy as np
from numpy.typing import NDArray
from image import Image


class Border:
    def __init__(self, border_fill_limit: float = 0.2):
        self.border_fill_limit = border_fill_limit

    # recebe uma imagem binarizada (preto e branco) e identifica as bordas com conteúdo
    # enquanto pelo menos 70% dos pixels na linha/coluna forem brancos, considera como borda
    def identify_borders(self, image: Image) -> tuple[int, int, int, int]:
        img = image.data
        if not image.binary:
            raise ValueError("Image must be binary (black and white) to identify borders")
        if not image.inversed:
            raise ValueError("Image must be inversed (white background, black content) to identify borders")
        h, w = img.shape
        top, bottom, left, right = 0, h - 1, 0, w - 1
        # print("topo")   
        # identificar topo
        for i in range(h):
            line = img[i, :]
            white_pixels = np.sum(line == 255)
            # print(white_pixels / w)
            if white_pixels / w < self.border_fill_limit:
                top = i
                break
        # print("top", top)
        # identificar bottom
        # print("bottom")
        for i in range(h - 1, -1, -1):
            line = img[i, :]
            white_pixels = np.sum(line == 255)
            # print(white_pixels / w)
            if white_pixels / w < self.border_fill_limit:
                bottom = i
                break
        # identificar left
        # print("bottom", bottom)
        # print("left")
        for j in range(w):
            column = img[:, j]
            white_pixels = np.sum(column == 255)
            # print(white_pixels / h)
            if white_pixels / h < self.border_fill_limit:
                left = j
                break
        # identificar right
        # print("left", left)
        # print("right")
        for j in range(w - 1, -1, -1):
            column = img[:, j]
            white_pixels = np.sum(column == 255)
            # print(white_pixels / h)
            if white_pixels / h < self.border_fill_limit:
                right = j
                break
        # print("right", right)
        return top, bottom, left, right

    def dynamic_cutter(self, img: Image) -> Image:
        top, bottom, left, right = self.identify_borders(img)
        return Image().set_data(img.data[top:bottom+1, left:right+1])
    
