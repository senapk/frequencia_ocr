from __future__ import annotations
import cv2
import numpy as np
import argparse
from numpy.typing import NDArray
from image import Image
from border import Border


class OCR:
    default_model_path = "raw/digits_knn.xml"
    def __init__(self, model_path: str = default_model_path):
        self.model_path = model_path
        self.model = cv2.ml.KNearest_load(self.model_path) # type: ignore
        self.debug: bool = False

    def set_debug(self, debug: bool) -> OCR:
        self.debug = debug
        return self
        
    def prepare_sample(self, img: Image) -> tuple[NDArray[np.float32], Image]:
        border = Border()
        img = border.dynamic_cutter_while_visible_border(img)
        if self.debug:
            print("Borda cortada:", img)
        img = border.dynamic_bounding_box(img)
        if self.debug:
            print("Bounding box aplicado:", img)
        img = border.centralize_and_pad(img)
        if self.debug:
            print("Centralizando e adicionando padding:", img)
        img = img.resize(20, 20)
        sample: NDArray[np.float32] = img.get_sample(400)
        return sample, img

    def predict_digit_from_image(self, img: Image) -> tuple[int, Image]:
        sample, img = self.prepare_sample(img)
        _, result, neighbours, dist = self.model.findNearest(sample, k=3)
        # normalizando distâncias
        dist = dist / 1000.0
        if self.debug:
            print("Predicted:", result[0][0], " Neighbours:", neighbours, " Distances:", dist)
        return int(result[0][0]), img

    def predict_digit_from_path(self, path: str) -> tuple[int, Image]:
        img = Image().load_from_file(path).binarize().inversion().cut_borders(2)
        return self.predict_digit_from_image(img)
def main():
    parser = argparse.ArgumentParser(description="OCR Digits")
    parser.add_argument("image", type=str, help="Path to the image file")
    args = parser.parse_args()
    Image.preview_zoom = 2
    ocr = OCR()
    digit, img = ocr.predict_digit_from_path(args.image)
    print(img)
    print(f"The recognized digit from {args.image} is: {digit}")

if __name__ == "__main__":
    main()