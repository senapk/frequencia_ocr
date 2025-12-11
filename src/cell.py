from __future__ import annotations
import cv2
import numpy as np
import os
from numpy.typing import NDArray
import base64


class Cell:
    NONE_MARKER = "?"

    def __init__(self):
        self.data: NDArray[np.uint8]
        self.info: None | str = None

    def clone(self) -> Cell:
        new_cell = Cell()
        new_cell.data = self.data.copy()
        return new_cell

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

    # if path has the format hash__v.ext, load v into info
    # if path has the format hash__?.ext, or any other format, load None to info
    def load_from_file(self, path: str) -> Cell:
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


    def save_to_file(self, file_path: str) -> None:
        cv2.imwrite(file_path, self.data)
        return None

    def set_data(self, data: NDArray[np.uint8]) -> Cell:
        self.data = data
        return self
    
    def __str__(self) -> str:
        _, buffer = cv2.imencode('.png', self.data)
        img_bytes = base64.b64encode(buffer).decode('utf-8')
        text = f"\033_Ga=T,f=100;{img_bytes}\033\\"
        return text
    
if __name__ == "__main__":
    cell = Cell().load_from_file("example.png")
    print(f"A imagem é : {cell}, o hash é {cell.hash()}")
    cell.info = "5"
    cell.save_to_folder("cache_folder")

    cell2 = Cell().load_from_file("cache_folder/" + cell.hash() + "__5.png")
    print(f"A imagem é : {cell2}, o hash é {cell2.hash()}, info = {cell2.info}")