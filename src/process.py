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
def cut_info(img: NDArray[np.uint8]) -> tuple[list[list[str]], list[list[Image]]]:
    lines = 25
    x_begin = 12
    y_begin = 268
    x_end = 246
    y_end = 1312
    cell_width = (x_end - x_begin) / 6
    cell_height = (y_end - y_begin) / lines
    # remove pasta temporária
    shutil.rmtree(cells_folder, ignore_errors=True)
    os.makedirs(cells_folder, exist_ok=True)

    infos: list[list[Image]] = [[] for _ in range(lines)]
    paths: list[list[str]] = [[] for _ in range(lines)]
    for i in range(lines):
        # x1line = x_begin
        # y1line = int(y_begin + i * cell_height)
        # x2line = int(x_begin + 6 * cell_width)
        # y2line = int(y_begin + (i + 1) * cell_height)
        # line_img = img[y1line:y2line, x1line:x2line]
        # cv2.imwrite(f"{cells_folder}/line_{i}.png", line_img)
        # print_image_in_terminal(f"{cells_folder}/line_{i}.png")
        for j in range(6):
            x1 = int(x_begin + j * cell_width)
            y1 = int(y_begin + i * cell_height)
            x2 = int(x_begin + (j + 1) * cell_width)
            y2 = int(y_begin + (i + 1) * cell_height)
            cell = Image().set_data(img[y1:y2, x1:x2])
            infos[i].append(cell)
            cv2.imwrite(f"{cells_folder}/{i}_{j}.png", cell.data)
            # paths[i].append(f"{cells_folder}/cell_row{i}_{j}.png")
            #subprocess.run(f"printf \"\033_Ga=T,f=100;%s\033\\\n\" \"$(base64 -w0 {cells_folder}/cell_row{i}_{j}.png)\"", shell=True)

    return paths, infos


def main():
    parser = argparse.ArgumentParser(description="Alinhar imagem usando marcadores ArUco.")
    parser.add_argument("--input", '-i', default="folhas/folha-00.png", help="Caminho para a imagem de entrada.")
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
    paths, images = cut_info(align_data)
    raw_cache_db = CacheDb(raw_cache_db_folder).load()
    bordered_cache_db = CacheDb(bordered_cache_db_folder).load()
    ocr_digit = OCR().set_debug(False)

    for i, row in enumerate(images):
        if args.qtd > 0 and i >= args.qtd:
            break
        for image in row:
            raw_cache_db.store_image(image)
            # print(image)
            image = image.binarize().inversion().cut_borders(2)
            bordered = Border(image).dynamic_cutter_while_visible_border().dynamic_bounding_box()
            bordered_cache_db.store_image(bordered.get_image())
            # print(bordered.get_image())
            digit, _ = ocr_digit.predict_from_bordered_image(bordered)
            write_ratio = bordered.get_image().written()
            w, h = bordered.get_image().data.shape
            if bordered.get_image().is_empty():
                continue
            print(f"{bordered.get_image().resize(40, 40)} {digit}, write ratio: {write_ratio:.0%}, size: {w}x{h}")
        # print("")

if __name__ == "__main__":
    main()