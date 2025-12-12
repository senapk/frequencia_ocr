import cv2
import numpy as np
from numpy.typing import NDArray
from image import Image
import shutil
import os

class SheetCutter:
    class Model:
        M1 = {
            "lines": 25,
            "columns": 6,
            "x_begin": 12,
            "y_begin": 268,
            "x_end": 246,
            "y_end": 1312
        }
        M2 = {
            "lines": 25,
            "columns": 6,
            "x_begin": 11,
            "y_begin": 242,
            "x_end": 246,
            "y_end": 1269
        }

    def __init__(self, cache_folder: str):
        self.cache_folder = cache_folder

    def cut_info(self, img: NDArray[np.uint8], model: dict[str, int]) -> tuple[list[list[Image]], list[list[str]]]:
        lines = model["lines"]
        x_begin = model["x_begin"]
        y_begin = model["y_begin"]
        x_end = model["x_end"]
        y_end = model["y_end"]
        cell_width = (x_end - x_begin) / model["columns"]
        cell_height = (y_end - y_begin) / lines
        # remove pasta temporária
        shutil.rmtree(self.cache_folder, ignore_errors=True)
        os.makedirs(self.cache_folder, exist_ok=True)

        infos: list[list[Image]] = [[] for _ in range(lines)]
        paths: list[list[str]] = [[] for _ in range(lines)]
        for i in range(lines):
            for j in range(6):
                x1 = int(x_begin + j * cell_width)
                y1 = int(y_begin + i * cell_height)
                x2 = int(x_begin + (j + 1) * cell_width)
                y2 = int(y_begin + (i + 1) * cell_height)
                cell = Image().set_data(img[y1:y2, x1:x2])
                infos[i].append(cell)
                cv2.imwrite(f"{self.cache_folder}/{i}_{j}.png", cell.data)
                
        return infos, paths