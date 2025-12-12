from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from image import Image
from typing import Callable


class Border:
    DEFAULT_BORDER_FILL_LIMIT = 0.2
    def __init__(self, image: Image):
        self.image: Image = image
        self.border_fill_limit = Border.DEFAULT_BORDER_FILL_LIMIT

    def get_image(self) -> Image:
        return self.image

    def set_border_fill_limit(self, limit: float) -> Border:
        self.border_fill_limit = limit
        return self

    # recebe uma imagem binarizada (preto e branco) e identifica as bordas com conteúdo
    # enquanto pelo menos 70% dos pixels na linha/coluna forem brancos, considera como borda
    def __identify_borders(self, image: Image, fn_continue: Callable[[NDArray[np.uint8]], bool]) -> tuple[int, int, int, int]:
        def fn(x: NDArray[np.uint8]) -> bool:
            return not fn_continue(x)
        
        img = image.data
        if not image.binary:
            raise ValueError("Image must be binary (black and white) to identify borders")
        if not image.inversed:
            raise ValueError("Image must be inversed (white background, black content) to identify borders")
        h, w = img.shape
        top, bottom, left, right = 0, h - 1, 0, w - 1
        # identificar topo
        for i in range(h):
            line = img[i, :]
            if fn(line):
                top = i
                break
        for i in range(h - 1, -1, -1):
            line = img[i, :]
            if fn(line):
                bottom = i
                break
        for j in range(w):
            column = img[:, j]
            if fn(column):
                left = j
                break
        for j in range(w - 1, -1, -1):
            column = img[:, j]
            if fn(column):
                right = j
                break
        return top, bottom, left, right

    # retorna a imagem cortada nas bordas onde ainda há conteúdo visível
    def dynamic_cutter_while_visible_border(self) -> Border:
        def fn(x: NDArray[np.uint8]) -> bool:
            qtd = np.sum(x == 255)
            amount = qtd / x.size
            return  amount >= self.border_fill_limit
        
        top, bottom, left, right = self.__identify_borders(self.image, fn)
        self.image = Image(self.image).set_data(self.image.data[top:bottom+1, left:right+1])
        return self
    
    # retorna a imagem cortada na bounding box do conteúdo
    def dynamic_bounding_box(self) -> Border:
        def fn(x: NDArray[np.uint8]) -> bool:
            qtd = np.sum(x == 0)
            return  qtd == x.size
        
        top, bottom, left, right = self.__identify_borders(self.image, fn)
        self.image = Image(self.image).set_data(self.image.data[top:bottom+1, left:right+1])
        return self
    
    # centraliza e faz o padding da imagem para ficar quadrada
    # e com fundo preto
    def centralize_and_pad(self, extra_pixels: int = 2) -> Border:
        h, w = self.image.data.shape
        size = max(h, w) + 2 * extra_pixels

        if self.image.inversed:
            background = 0
        else:
            background = 255

        # cria canvas quadrado com a cor desejada
        padded = np.ones((size, size), dtype=np.uint8) * background

        y_offset = (size - h) // 2
        x_offset = (size - w) // 2

        padded[y_offset:y_offset + h, x_offset:x_offset + w] = self.image.data

        self.image = Image(self.image).set_data(padded)
        return self
