from __future__ import annotations
import cv2
import numpy as np
import os
from numpy.typing import NDArray
import base64


class Image:
    NONE_MARKER = "?"

    def __init__(self):
        self.data: NDArray[np.uint8] = np.array([], dtype=np.uint8)
        self.info: None | str = None
        self.binary: bool = False
        self.inversed: bool = False

    def hash(self) -> str:
        value = abs(hash(self.data.tobytes()))
        choices = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTWXYZ0123456789"
        size = len(choices)
        response: str = ""
        while value > 0:
            left = value % size
            response = choices[left] + response
            value = value // size
        return response
    
    def set_data(self, data: NDArray[np.uint8]) -> Image:
        self.data = data
        return self
    
    def set_info(self, info: str) -> Image:
        self.info = info
        return self

    def set_binary(self, binary: bool = True) -> Image:
        self.binary = binary
        return self

    def set_inversed(self, inversed: bool = True) -> Image:
        self.inversed = inversed
        return self

    def get_info(self) -> None | str:
        return self.info
    
    def get_data(self) -> NDArray[np.uint8]:
        return self.data

    # if path has the format hash__v.ext, load v into info
    # if path has the format hash__?.ext, or any other format, load None to info
    def load_from_file(self, path: str) -> Image:
        self.data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        filename = path.split(".")[-2]
        if len(filename) > 3 and (filename[-3:-2] =="__"):
            self.info = None if filename[-1] == self.NONE_MARKER else filename[-1]
        return self
    
    # save to cache folder using hash and value
    def save_to_folder(self, folder: str):
        path = os.path.join(folder, self.hash())
        info = self.info if self.info is not None else self.NONE_MARKER
        path += f"__{info}.png"
        self.save_to_file(path)

    # return a new Cell with borders cut
    def cut_borders(self, pixels_count: int) -> Image:
        h, w = self.data.shape
        return Image().set_data(self.data[pixels_count:h-pixels_count, pixels_count:w-pixels_count])

    def save_to_file(self, file_path: str) -> None:
        cv2.imwrite(file_path, self.data)
        return None
    
    def binarize_and_invert(self) -> Image:
        _, binary = cv2.threshold(self.data, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return Image().set_data(binary).set_inversed(True).set_binary(True)
    
    def resize(self, dx: int, dy: int) -> Image:
        return Image().set_data(cv2.resize(self.data, (dx, dy), interpolation=cv2.INTER_AREA))

    # transform in a array
    def get_sample(self, points: int = 400) -> NDArray[np.float32]:
        return self.data.reshape(1, points).astype(np.float32)

    def __str__(self) -> str:
        _, buffer = cv2.imencode('.png', self.data)
        img_bytes = base64.b64encode(buffer).decode('utf-8')
        text = f"\033_Ga=T,f=100;{img_bytes}\033\\"
        return text
    
if __name__ == "__main__":
    cell = Image().load_from_file("example.png")
    print(f"A imagem é : {cell}, o hash é {cell.hash()}")
    cell.info = "5"
    cache_folder = "cache_folder"
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    cell.save_to_folder(cache_folder)

    cell2 = Image().load_from_file(os.path.join(cache_folder, cell.hash() + "__5.png"))
    print(f"A imagem é : {cell2}, o hash é {cell2.hash()}, info = {cell2.info}")