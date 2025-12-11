import cv2
import numpy as np
import argparse
from numpy.typing import NDArray
from src.image import Image
from align import Align
from filter import Filter
from src.border import Border
from color import Color

HEIGHT = 1400
WIDTH = 1000

cells_folder = "cells"
preview_size = 30
default_border = 2
border_fill_limit = 0.2
model_path = "raw/digits_knn.xml"


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
        x1line = x_begin
        y1line = int(y_begin + i * cell_height)
        x2line = int(x_begin + 6 * cell_width)
        y2line = int(y_begin + (i + 1) * cell_height)
        line_img = img[y1line:y2line, x1line:x2line]
        cv2.imwrite(f"{cells_folder}/line_{i}.png", line_img)
        # print_image_in_terminal(f"{cells_folder}/line_{i}.png")
        for j in range(6):
            x1 = int(x_begin + j * cell_width)
            y1 = int(y_begin + i * cell_height)
            x2 = int(x_begin + (j + 1) * cell_width)
            y2 = int(y_begin + (i + 1) * cell_height)
            cell = Image().set_data(img[y1:y2, x1:x2])
            row.append(cell)
            cv2.imwrite(f"{cells_folder}/cell_row{i}_{j}.png", cell.data)
            paths[i].append(f"{cells_folder}/cell_row{i}_{j}.png")
            #subprocess.run(f"printf \"\033_Ga=T,f=100;%s\033\\\n\" \"$(base64 -w0 {cells_folder}/cell_row{i}_{j}.png)\"", shell=True)
        infos[i] = row

    return paths, infos




def predict_digit(img: Image, model: cv2.ml_KNearest) -> int:
    print("orig:", img)

    img = img.cut_borders(default_border)
    # img = cv2.equalizeHist(img)
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    # binarização otsu (scanner de documentos)
    _, img.data = cv2.threshold(img.data, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # binarização adaptativa (fotos com iluminação irregular)
    # img = cv2.adaptiveThreshold( img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10 ) # type: ignore
    print("bin:", img)

    # img = cut_borders(img, border_cut)
    border = Border(border_fill_limit)
    img = border.dynamic_cutter(img)
    print("border:", img)

    # erodir para remover ruídos
    # print(" ,erode:", end="")
    # kernel = np.ones((2,2), np.uint8)
    # img = cv2.erode(img, kernel, iterations=1)
    # preview_square(img)

    # dilatar para reforçar traços
    # kernel = np.ones((2,2), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=1)
    # print_preview("dilate:", img)
    
    img = img.resize(20, 20)
    sample: NDArray[np.float32] = img.get_sample(400)

    ret, result, neighbours, dist = model.findNearest(sample, k=3) # type: ignore
    return int(result[0][0])


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
    filter = Filter()
    img = filter.otsu(img, args.filter)
    paths, images = cut_info(img)

     # type: ignore
    for i, row in enumerate(images):
        if args.qtd > 0 and i >= args.qtd:
            break
        print(f"Linha {i + 1}: ")
        for cell_path in row:
            digit = predict_digit(cell_path, model)
            print(f"{digit}, ", end="\n")
        print("")
    