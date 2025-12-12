import cv2
import numpy as np
import argparse
from numpy.typing import NDArray
from image import Image
from align import Align
from filtering import Filtering
from border import Border
from color import Color
from ocr import OCR
from cache_db import CacheDb
from sheet_cutter import SheetCutter
import shutil
import os
import hashlib


HEIGHT = 1400
WIDTH = 1000
raw_cache_db_folder = "cache_digits_raw"
bordered_cache_db_folder = "cache_digits_processed"
cells_folder = "cache_temp"


# retorna a lista de listas de caminhos das células recortadas
# retorna a lista com as células recortadas


def main():
    parser = argparse.ArgumentParser(description="Alinhar imagem usando marcadores ArUco.")
    parser.add_argument("--input", '-i', default="folhas/m2-1.png", help="Caminho para a imagem de entrada.")
    parser.add_argument("--sheet_model", '-m', type=int, default=2, help="Modelo da folha (1 ou 2).")
    parser.add_argument("--align", '-a', default="folhas/alinhada.png", help="Caminho para salvar a imagem alinhada.")
    parser.add_argument("--filter", '-f', default="folhas/filtrada.png", help="Caminho para salvar a imagem filtrada.")
    parser.add_argument("--qtd", type=int, default=0, help="Quantidade de linhas a processar (0 = todas).")
    args = parser.parse_args()

    aligner = Align(WIDTH, HEIGHT)
    image = Image().load_from_file(args.input)
    # print("Imagem de entrada:", image_input)
    print("Hash da imagem de entrada:", image.hash())

    align_data = aligner.aruco(image.data)
    image_aligned = Image().set_data(align_data)
    image_aligned.save_to_file(args.align)
    print("Imagem alinhada salva em", Color.green(args.align))
    print("Hash da imagem alinhada:", image_aligned.hash())

    # filtering = Filtering()
    # img = filtering.otsu(img, args.filter)
    
    sheet_cutter = SheetCutter(cells_folder)
    model = SheetCutter.Model.M1 if args.sheet_model == 1 else SheetCutter.Model.M2
    images, _ = sheet_cutter.cut_info(align_data, model)

    raw_cache_db = CacheDb(raw_cache_db_folder).load()
    bordered_cache_db = CacheDb(bordered_cache_db_folder).load()
    ocr_digit = OCR().set_debug(False)

    
    for i, row in enumerate(images):
        if args.qtd > 0 and i >= args.qtd:
            break
        for image_origin in row:
            raw_cache_db.store_image(image_origin)
            # print(image)
            image = image_origin.binarize(gaussian=False).inversion().cut_borders(2)
            bordered_before_paddding = Border(image).dynamic_cutter_while_visible_border().dynamic_bounding_box()
            write_ratio = bordered_before_paddding.get_image().written_ratio()
            if bordered_before_paddding.get_image().is_empty(): # check empty before padding
                continue
            bordered_padded = bordered_before_paddding.centralize_and_pad()
            bordered_cache_db.store_image(bordered_padded.get_image())
            digit, _ = ocr_digit.predict_from_bordered_image(bordered_padded)
            h, w = bordered_before_paddding.get_image().get_dim()
            print(f"{image_origin}{bordered_before_paddding.get_image().resize(40, 40)} {digit}, write ratio: {write_ratio:.0%}, size: {w}x{h}")
        # print("")

if __name__ == "__main__":
    main()