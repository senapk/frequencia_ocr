import cv2
import numpy as np
import argparse
import subprocess

HEIGHT = 1400
WIDTH = 1000

cells_folder = "cells"

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

    dst = np.float32([
        [0, 0],
        [W, 0],
        [W, H],
        [0, H]
    ])

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
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # fazer contornos
    # gray = cv2.Canny(gray, 50, 150, apertureSize=3)

    cv2.imwrite(path, gray)
    print("Imagem filtrada salva em", green(path))
    return gray

def cut_info(img):
    lines = 25
    x_begin = 20
    y_begin = 286
    cell_width = (250 - 20) / 6
    cell_height = (1313 - 286) / lines
    infos: list[list[np.ndarray]] = [[] for _ in range(lines)]
    for i in range(lines):
        row = []
        print("")
        x1line = x_begin
        y1line = int(y_begin + i * cell_height)
        x2line = int(x_begin + 6 * cell_width)
        y2line = int(y_begin + (i + 1) * cell_height)
        line_img = img[y1line:y2line, x1line:x2line]
        cv2.imwrite(f"{cells_folder}/line_{i}.jpg", line_img)
        subprocess.run(f"printf \"\033_Ga=T,f=100;%s\033\\\n\" \"$(base64 -w0 {cells_folder}/line_{i}.jpg)\"", shell=True)
        print("")
        for j in range(6):
            x1 = int(x_begin + j * cell_width)
            y1 = int(y_begin + i * cell_height)
            x2 = int(x_begin + (j + 1) * cell_width)
            y2 = int(y_begin + (i + 1) * cell_height)
            cell = img[y1:y2, x1:x2]
            row.append(cell)
            cv2.imwrite(f"{cells_folder}/cell_row{i}_{j}.jpg", cell)
            #subprocess.run(f"printf \"\033_Ga=T,f=100;%s\033\\\n\" \"$(base64 -w0 {cells_folder}/cell_row{i}_{j}.jpg)\"", shell=True)
        infos[i] = row

    return infos

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alinhar imagem usando marcadores ArUco.")
    parser.add_argument("--input", '-i', default="folha.jpg", help="Caminho para a imagem de entrada.")
    parser.add_argument("--align", '-a', default="alinhada.jpg", help="Caminho para salvar a imagem alinhada.")
    parser.add_argument("--filter", '-f', default="filtrada.jpg", help="Caminho para salvar a imagem filtrada.")
    args = parser.parse_args()

    img = alinhar(args.input, args.align)
    filtrar(img, args.filter)
    cut_info(img)

print("Folha corrigida gerada com sucesso.")
