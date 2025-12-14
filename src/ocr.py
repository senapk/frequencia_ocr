from __future__ import annotations
import cv2
import numpy as np
import argparse
from numpy.typing import NDArray
from image import Image
from border import Border
from filter_abc import ImageFilter


class OCR:
    # default_model_path = "raw/digits_knn.xml"
    models: dict[int, str] = {
        20: "raw/mnist_knn_20x20.yml",
        28: "raw/mnist_knn_28x28.yml"
    }
    
    def __init__(self, model_size: int = 20):
        self.model_path = self.models[model_size]
        self.size = model_size
        self.model = cv2.ml.KNearest_load(self.model_path) # type: ignore
        self.debug: bool = False

    def set_debug(self, debug: bool) -> OCR:
        self.debug = debug
        return self
        
    def prepare_sample(self, img: Image) -> tuple[NDArray[np.float32], Image]:
        img = img.resize(self.size, self.size)
        sample: NDArray[np.float32] = img.get_sample(self.size * self.size)
        return sample, img

    def predict_from_raw_image(self, img: Image) -> tuple[int, Image]:
        border = Border(img)
        border.dynamic_cutter_while_visible_border()
        if self.debug:
            print("Borda cortada:", border.get_image())
        border.dynamic_bounding_box()
        if self.debug:
            print("Bounding box aplicado:", border.get_image())
        border.centralize_and_pad(self.size)
        if self.debug:
            print("Centralizando e adicionando padding:", border.get_image())
        return self.predict_from_filtered_image(border)

    def predict_from_filtered_image(self, filtered: ImageFilter) -> tuple[int, Image]:
        sample, img = self.prepare_sample(filtered.get_image())
        _, result, neighbours, dist = self.model.findNearest(sample, k=3)
        # normalizando distâncias
        dist = dist / 1000.0
        if self.debug:
            print("Predicted:", result[0][0], " Neighbours:", neighbours, " Distances:", dist)
        return int(result[0][0]), img

    def predict_from_raw_path(self, path: str) -> tuple[int, Image]:
        img = Image().load_from_file(path).binarize().inversion().cut_borders(2)
        return self.predict_from_raw_image(img)
def main():
    parser = argparse.ArgumentParser(description="OCR Digits")
    parser.add_argument("image", type=str, help="Path to the image file")
    args = parser.parse_args()
    Image.preview_zoom = 2
    ocr = OCR()
    digit, img = ocr.predict_from_raw_path(args.image)
    print(img)
    print(f"The recognized digit from {args.image} is: {digit}")

if __name__ == "__main__":
    main()