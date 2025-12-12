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


HEIGHT = 1400
WIDTH = 1000
cache_db_folder = "cache_digits"


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
    infos: list[list[Image]] = [[] for _ in range(lines)]

    paths: list[list[str]] = [[] for _ in range(lines)]
    for i in range(lines):
        row: list[Image] = []
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
            row.append(cell)
            # cv2.imwrite(f"{cells_folder}/cell_row{i}_{j}.png", cell.data)
            # paths[i].append(f"{cells_folder}/cell_row{i}_{j}.png")
            #subprocess.run(f"printf \"\033_Ga=T,f=100;%s\033\\\n\" \"$(base64 -w0 {cells_folder}/cell_row{i}_{j}.png)\"", shell=True)
        infos[i] = row

    return paths, infos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alinhar imagem usando marcadores ArUco.")
    parser.add_argument("--input", '-i', default="folhas/folha-00.png", help="Caminho para a imagem de entrada.")
    parser.add_argument("--align", '-a', default="folhas/alinhada.png", help="Caminho para salvar a imagem alinhada.")
    parser.add_argument("--filter", '-f', default="folhas/filtrada.png", help="Caminho para salvar a imagem filtrada.")
    parser.add_argument("--qtd", type=int, default=0, help="Quantidade de linhas a processar (0 = todas).")
    parser.add_argument("--preview", '-p', type=int, default=30, help="Tamanho do preview das células.")
    args = parser.parse_args()

    preview_size = args.preview

    aligner = Align(WIDTH, HEIGHT)
    img = aligner.aruco(args.input, args.align)
    # filtering = Filtering()
    # img = filtering.otsu(img, args.filter)
    paths, images = cut_info(img)
    cache_db = CacheDb(cache_db_folder)
    cache_db.load_folder()
    ocr = OCR().set_debug(False)

    for i, row in enumerate(images):
        if args.qtd > 0 and i >= args.qtd:
            break
        for image in row:
            cache_db.store_image(image)
            print(image)
            image = image.binarize().inversion().cut_borders(2)
            print(image)
            digit, img = ocr.predict_digit_from_image(image)
            # print(f"{digit}, ", end="\n")
        print("")
    