import cv2
import numpy as np
import argparse
from numpy.typing import NDArray
from image import Image
from border import Border


class Ocr:
    model_path = "raw/digits_knn.xml"

    def __init__(self):
        self.model = cv2.ml.KNearest_load(self.model_path) # type: ignore

    def prepare_sample(self, img: Image) -> NDArray[np.float32]:
        border = Border()
        img = border.dynamic_cutter(img)
        print(img)
        img = img.resize(20, 20)
        sample: NDArray[np.float32] = img.get_sample(400)
        return sample

    def predict_digit(self, img: Image) -> int:
        sample = self.prepare_sample(img)
        _, result, neighbours, dist = self.model.findNearest(sample, k=3)
        print("Predicted:", result[0][0], " Neighbours:", neighbours, " Distances:", dist)
        return int(result[0][0])

def main():
    parser = argparse.ArgumentParser(description="OCR Digits")
    parser.add_argument("image", type=str, help="Path to the image file")
    args = parser.parse_args()

    ocr = Ocr()
    img = Image().load_from_file(args.image).binarize_and_invert()
    digit = ocr.predict_digit(img)
    print(img)
    print("The recognized digit is:", digit)

if __name__ == "__main__":
    main()