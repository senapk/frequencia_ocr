import cv2
import numpy as np
import argparse
import subprocess
import base64

HEIGHT = 1400
WIDTH = 1000

cells_folder = "cells"

model_path = "raw/digits_knn.xml"

def print_image(image_path: str, end="\n"):
    # ler e converter para base64 (string sem quebras)
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    # imprimir no terminal (WezTerm/Kitty)
    print(f"\033_Ga=T,f=100;{data}\033\\", end=end)

def print_raw_img(img, end="\n"):
    cv2.imwrite("temp_cell.jpg", img)
    print_image("temp_cell.jpg", end=end)

def green(text: str) -> str:
    return f"\033[92m{text}\033[0m"
# Carregar imagem

def alinhar(input: str, output: str):
    img = cv2.imread(input)

    # Configurar ArUco
    aruco = cv2.aruco
    dict_ = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dict_, params)

    # Detectar marcadores
    corners, ids, _ = detector.detectMarkers(img)

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
    M = cv2.getPerspectiveTransform(pts, dst)
    corrigida = cv2.warpPerspective(img, M, (W, H))

    cv2.imwrite(output, corrigida)
    print("Imagem alinhada salva em", green(output))
    return corrigida

def filtrar(img, path: str):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # aumentar contraste
    gray = cv2.equalizeHist(gray)

    # remover ruído
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # fazer contornos
    # gray = cv2.Canny(gray, 50, 150, apertureSize=3)

    cv2.imwrite(path, gray)
    print("Imagem filtrada salva em", green(path))
    return gray

def cut_info(img) -> list[list[str]]:
    lines = 25
    x_begin = 20
    y_begin = 286
    cell_width = (250 - 20) / 6
    cell_height = (1313 - 286) / lines
    infos: list[list[np.ndarray]] = [[] for _ in range(lines)]

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

def preview_square(img):
    view_size = 50
    temp_view = cv2.resize(img, (view_size, view_size), interpolation=cv2.INTER_AREA)
    print_raw_img(temp_view, "")

def auto_cut_borders(img):
    # encontrar pixels não-zero (parte do dígito)
    ys, xs = np.where(img > 0)

    # bounding box mínimo
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    digit = img[y1:y2+1, x1:x2+1]
    return digit

def cut_borders(img, pixels: int):
    h, w = img.shape
    return img[pixels:h-pixels, pixels:w-pixels]

def predict_digit(img_path, model):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print("orig:", end="")
    preview_square(img)

    print(", border:", end="")
    img = cut_borders(img, 4)
    preview_square(img)

    # img = cv2.equalizeHist(img)
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    # binarização otsu (scanner de documentos)
    #_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # binarização adaptativa (fotos com iluminação irregular)
    img = cv2.adaptiveThreshold( img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10 )
    print(", bin:", end="")
    preview_square(img)
    
    # erodir para remover ruídos
    print(" ,erode:", end="")
    kernel = np.ones((2,2), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    preview_square(img)

    # dilatar para reforçar traços
    print(", dilate:", end="")
    kernel = np.ones((2,2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    preview_square(img)
    
    img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
    sample = img.reshape(1, 400).astype(np.float32)

    ret, result, neighbours, dist = model.findNearest(sample, k=3)
    return int(result[0][0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alinhar imagem usando marcadores ArUco.")
    parser.add_argument("--input", '-i', default="folha.jpg", help="Caminho para a imagem de entrada.")
    parser.add_argument("--align", '-a', default="alinhada.jpg", help="Caminho para salvar a imagem alinhada.")
    parser.add_argument("--filter", '-f', default="filtrada.jpg", help="Caminho para salvar a imagem filtrada.")
    parser.add_argument("--qtd", type=int, default=0, help="Quantidade de linhas a processar (0 = todas).")
    args = parser.parse_args()

    img = alinhar(args.input, args.align)
    # img = filtrar(img, args.filter)
    paths = cut_info(img)

    model = cv2.ml.KNearest_load(model_path)
    for i, row in enumerate(paths):
        if args.qtd > 0 and i >= args.qtd:
            break
        print(f"Linha {i + 1}: ")
        for cell_path in row:
            digit = predict_digit(cell_path, model)
            print(f"{digit}, ", end="\n")
        print("")
    