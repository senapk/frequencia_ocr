import cv2
import numpy as np
from numpy.typing import NDArray
from image import Image
import shutil
import os

class SheetCutter:
    class Model:
        M1: dict[str, float] = {
            "lines": 25,
            "columns": 6,
            "x_begin": 12,
            "y_begin": 268,
            "x_end": 246,
            "y_end": 1312
        }
        M2: dict[str, float] = {
            "lines": 25,
            "columns": 6,
            "x_begin": 0.0134,
            "y_begin": 0.174,
            "x_end": 0.246,
            "y_end": 0.9064285714285715
        }

    def __init__(self, cache_folder: str):
        self.cache_folder = cache_folder

    def cut_info(self, img: Image, model: dict[str, float]) -> tuple[list[list[Image]], list[list[str]]]:
        h, w = img.get_h_w()
        
        lines = int(model["lines"])
        x_ini: float = model["x_begin"] * w
        y_ini: float = model["y_begin"] * h
        x_end: float = model["x_end"] * w
        y_end: float = model["y_end"] * h
        cell_w: float = (x_end - x_ini) / model["columns"]
        cell_h: float = (y_end - y_ini) / lines
        # print(f"Cell size: {cell_w} x {cell_h}")
        # remove pasta temporária
        shutil.rmtree(self.cache_folder, ignore_errors=True)
        os.makedirs(self.cache_folder, exist_ok=True)

        infos: list[list[Image]] = [[] for _ in range(lines)]
        paths: list[list[str]] = [[] for _ in range(lines)]
        for i in range(lines):
            for j in range(6):
                x1 = int(x_ini + j * cell_w)
                y1 = int(y_ini + i * cell_h)
                x2 = x1 + int(cell_w)
                y2 = y1 + int(cell_h)
                cell = Image(img).set_data(img.data[y1:y2, x1:x2])
                infos[i].append(cell)
                cv2.imwrite(f"{self.cache_folder}/{i}_{j}.png", cell.data)
                
        return infos, paths