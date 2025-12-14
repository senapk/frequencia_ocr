from __future__ import annotations
import cv2
import numpy as np
import os
from numpy.typing import NDArray
import base64
import argparse
import hashlib

class Image:
    NONE_MARKER = "?"
    preview_zoom: float | None = None

    def __init__(self, other: None | Image = None):
        if other is None:
            self.data: NDArray[np.uint8] = np.array([], dtype=np.uint8)
            self.info: None | str = None
            self.binary: bool = False
            self.inversed: bool = False
        else:
            self.data = other.data.copy()
            self.info = other.info
            self.binary = other.binary
            self.inversed = other.inversed

    def hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.data.tobytes())
        return h.hexdigest()[:16]
    
    def set_data(self, data: NDArray[np.uint8]) -> Image:
        self.data = data
        return self
    
    # percentage of white pixels
    def written_ratio(self) -> float:
        if self.data.size == 0:
            return 0.0
        ref_value = 255 if self.inversed else 0
        total_pixels = self.data.size
        white_pixels = np.sum(self.data == ref_value)
        return white_pixels / total_pixels
    
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
        self.data = cv2.imread(path)
        return self
    
    # save to cache folder using hash and value
    def save_to_folder(self, folder: str):
        path = os.path.join(folder, self.hash()) + ".png"
        self.save_to_file(path)

    # return a new Cell with borders cut
    def cut_borders(self, pixels_count: int) -> Image:
        h, w = self.get_h_w()
        return Image(self).set_data(self.data[pixels_count:h-pixels_count, pixels_count:w-pixels_count])

    def save_to_file(self, file_path: str) -> None:
        cv2.imwrite(file_path, self.data)
        return None
    
    def binarize(self, gaussian: bool = True) -> Image:
        # se estiver em RGB/BGR, converter para escala de cinza primeiro
        img = self.data
        if self.data.ndim != 2:  # RGB/BGR/RGBA
            img = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
        if gaussian:
            img = cv2.GaussianBlur(img, (3,3), 0)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image(self).set_data(binary).set_binary(True)
    
    def inversion(self) -> Image:
        inverted = cv2.bitwise_not(self.data)
        return Image(self).set_data(inverted).set_inversed(not self.inversed)
    
    
    def get_h_w(self) -> tuple[int, int]:
        try:
            return (self.data.shape[0], self.data.shape[1])
        except IndexError as _:
            return (0, 0)

    # empty if dimensions are too small or written ratio < 2%
    def is_empty(self) -> bool:
        h, w = self.get_h_w()
        if h < 5 or w < 5:
            return True
        if self.written_ratio() < 0.02:
            return True
        return False

    def resize(self, dx: int, dy: int) -> Image:
        h, w = self.get_h_w()
        if h == dy and w == dx:
            return self
        return Image(self).set_data(cv2.resize(self.data, (dx, dy), interpolation=cv2.INTER_AREA))

    # transform in a array
    def get_sample(self, points: int) -> NDArray[np.float32]:
        return self.data.reshape(1, points).astype(np.float32)

    def __str__(self) -> str:
        data = self.data
        if self.preview_zoom is not None:
            h, w = self.get_h_w()
            w = int(w * self.preview_zoom)
            h = int(h * self.preview_zoom)
            data = cv2.resize(data, (h, w), interpolation=cv2.INTER_AREA)
 
        _, buffer = cv2.imencode('.png', data)
        img_bytes = base64.b64encode(buffer).decode('utf-8')
        text = f"\033_Ga=T,f=100;{img_bytes}\033\\"
        return text
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Test")
    parser.add_argument("image", type=str, help="Path to the image file")
    args = parser.parse_args()
    Image.preview_zoom = 2
    img = Image().load_from_file(args.image)
    print(img)
    img = img.binarize()
    print(img)