import cv2
import numpy as np
import argparse
import base64
import tempfile
from numpy.typing import NDArray


HEIGHT = 1400
WIDTH = 1000

cells_folder = "cells"
preview_size = 30
border_cut = 2
border_fill_limit = 0.2
model_path = "raw/digits_knn.xml"


def print_image(image_path: str, end: str="\n"):
    # ler e converter para base64 (string sem quebras)
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    # imprimir no terminal (WezTerm/Kitty)
    print(f"\033_Ga=T,f=100;{data}\033\\", end=end)

def print_raw_img(img: NDArray[np.uint8], end: str="\n"):
    temp_jpg_file = tempfile.gettempdir() + "/temp_cell.jpg"
    cv2.imwrite(temp_jpg_file, img)
    print_image(temp_jpg_file, end=end)

def green(text: str) -> str:
    return f"\033[92m{text}\033[0m"
# Carregar imagem

def alinhar(input: str, output: str):
    img = cv2.imread(input)

    # Configurar ArUco
    aruco = cv2.aruco
    dict_ = aruco.getPredefinedDictionary(aruco.DICT_4X4_50) # type: ignore
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dict_, params) # type: ignore

    # Detectar marcadores
    corners, ids, _ = detector.detectMarkers(img) # type: ignore

    if ids is None or len(ids) < 4:
        raise Exception("Nem todos os 4 ArUcos foram detectados.")

    ids = ids.flatten()

    # Array que guardará os centros ordenados como [TL, TR, BR, BL]
    pts = np.zeros((4, 2), dtype=np.float32)

    for c, id in zip(corners, ids):
        c = c[0]

        if id == 0:  # TL
            pts[0] = c[0]  # canto superior esquerdo do marcador 0
        elif id == 1:  # TR
            pts[1] = c[1]  # canto superior direito
        elif id == 3:  # BR
            pts[2] = c[2]  # canto inferior direito
        elif id == 2:  # BL
            pts[3] = c[3]  # canto inferior esquerdo

    # tamanho final
    W, H = WIDTH, HEIGHT

    dst = np.float32([ [0, 0], [W, 0], [W, H], [0, H] ]) # type: ignore

    # Matriz de transformação e warp
    M = cv2.getPerspectiveTransform(pts, dst) # type: ignore
    corrigida = cv2.warpPerspective(img, M, (W, H)) # type: ignore

    cv2.imwrite(output, corrigida)
    print("Imagem alinhada salva em", green(output))
    return corrigida

def filtrar(img: NDArray[np.uint8], path: str):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # aumentar contraste
    # img = cv2.equalizeHist(img)

    # remover ruído
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # binarização otsu (scanner de documentos)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # fazer contornos
    # gray = cv2.Canny(gray, 50, 150, apertureSize=3)

    cv2.imwrite(path, img)
    print("Imagem filtrada salva em", green(path))
    return img

def cut_info(img: NDArray[np.uint8]) -> list[list[str]]:
    lines = 25
    x_begin = 12
    y_begin = 268
    x_end = 246
    y_end = 1312
    cell_width = (x_end - x_begin) / 6
    cell_height = (y_end - y_begin) / lines
    infos: list[list[NDArray[np.uint8]]] = [[] for _ in range(lines)]

    paths: list[list[str]] = [[] for _ in range(lines)]
    for i in range(lines):
        row = []
        x1line = x_begin
        y1line = int(y_begin + i * cell_height)
        x2line = int(x_begin + 6 * cell_width)
        y2line = int(y_begin + (i + 1) * cell_height)
        line_img = img[y1line:y2line, x1line:x2line]
        cv2.imwrite(f"{cells_folder}/line_{i}.jpg", line_img)
        # print_image_in_terminal(f"{cells_folder}/line_{i}.jpg")
        for j in range(6):
            x1 = int(x_begin + j * cell_width)
            y1 = int(y_begin + i * cell_height)
            x2 = int(x_begin + (j + 1) * cell_width)
            y2 = int(y_begin + (i + 1) * cell_height)
            cell = img[y1:y2, x1:x2]
            row.append(cell)
            cv2.imwrite(f"{cells_folder}/cell_row{i}_{j}.jpg", cell)
            paths[i].append(f"{cells_folder}/cell_row{i}_{j}.jpg")
            #subprocess.run(f"printf \"\033_Ga=T,f=100;%s\033\\\n\" \"$(base64 -w0 {cells_folder}/cell_row{i}_{j}.jpg)\"", shell=True)
        infos[i] = row

    return paths

# argumento variadico que pode ser string ou NDArray
def print_preview(*args: str | NDArray[np.uint8]):
    for elem in args:
        if isinstance(elem, str):
            print(f"{elem}", end="")
        else:
            img = elem
            temp_view = cv2.resize(img, (preview_size, preview_size), interpolation=cv2.INTER_AREA)
            print_raw_img(temp_view, "")


def cut_borders(img: NDArray[np.uint8], pixels: int) -> NDArray[np.uint8]:
    h, w = img.shape
    return img[pixels:h-pixels, pixels:w-pixels]

# recebe uma imagem binarizada (preto e branco) e identifica as bordas com conteúdo
# enquanto pelo menos 70% dos pixels na linha/coluna forem brancos, considera como borda
def identify_borders(img: NDArray[np.uint8]) -> tuple[int, int, int, int]:
    h, w = img.shape
    top, bottom, left, right = 0, h - 1, 0, w - 1
    # print("topo")   
    # identificar topo
    for i in range(h):
        line = img[i, :]
        white_pixels = np.sum(line == 255)
        # print(white_pixels / w)
        if white_pixels / w < border_fill_limit:
            top = i
            break
    # print("top", top)
    # identificar bottom
    # print("bottom")
    for i in range(h - 1, -1, -1):
        line = img[i, :]
        white_pixels = np.sum(line == 255)
        # print(white_pixels / w)
        if white_pixels / w < border_fill_limit:
            bottom = i
            break
    # identificar left
    # print("bottom", bottom)
    # print("left")
    for j in range(w):
        column = img[:, j]
        white_pixels = np.sum(column == 255)
        # print(white_pixels / h)
        if white_pixels / h < border_fill_limit:
            left = j
            break
    # identificar right
    # print("left", left)
    # print("right")
    for j in range(w - 1, -1, -1):
        column = img[:, j]
        white_pixels = np.sum(column == 255)
        # print(white_pixels / h)
        if white_pixels / h < border_fill_limit:
            right = j
            break
    # print("right", right)
    return top, bottom, left, right

def cut_borders_dynamic(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    top, bottom, left, right = identify_borders(img)
    return img[top:bottom+1, left:right+1]

def predict_digit(img_path: str, model: cv2.ml_KNearest) -> int:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print_preview("orig:", img)

    img = cut_borders(img, border_cut)
    # img = cv2.equalizeHist(img)
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    # binarização otsu (scanner de documentos)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # binarização adaptativa (fotos com iluminação irregular)
    # img = cv2.adaptiveThreshold( img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10 ) # type: ignore
    print_preview("bin:", img)

    # img = cut_borders(img, border_cut)
    img = cut_borders_dynamic(img)
    print_preview("border:", img)

    # erodir para remover ruídos
    # print(" ,erode:", end="")
    # kernel = np.ones((2,2), np.uint8)
    # img = cv2.erode(img, kernel, iterations=1)
    # preview_square(img)

    # dilatar para reforçar traços
    kernel = np.ones((2,2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    print_preview("dilate:", img)
    
    img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
    sample = img.reshape(1, 400).astype(np.float32)

    ret, result, neighbours, dist = model.findNearest(sample, k=3) # type: ignore
    return int(result[0][0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alinhar imagem usando marcadores ArUco.")
    parser.add_argument("--input", '-i', default="folha.jpg", help="Caminho para a imagem de entrada.")
    parser.add_argument("--align", '-a', default="alinhada.jpg", help="Caminho para salvar a imagem alinhada.")
    parser.add_argument("--filter", '-f', default="filtrada.jpg", help="Caminho para salvar a imagem filtrada.")
    parser.add_argument("--qtd", type=int, default=0, help="Quantidade de linhas a processar (0 = todas).")
    parser.add_argument("--preview", '-p', type=int, default=30, help="Tamanho do preview das células.")
    args = parser.parse_args()

    preview_size = args.preview

    img = alinhar(args.input, args.align)
    filtrar(img, args.filter)
    paths = cut_info(img)

    model = cv2.ml.KNearest_load(model_path) # type: ignore
    for i, row in enumerate(paths):
        if args.qtd > 0 and i >= args.qtd:
            break
        print(f"Linha {i + 1}: ")
        for cell_path in row:
            digit = predict_digit(cell_path, model)
            print(f"{digit}, ", end="\n")
        print("")
    