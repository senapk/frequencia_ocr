import cv2
import numpy as np
from color import Color
from numpy.typing import NDArray

class Align:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height


    # A folha possui 4 ArUcos nos cantos para alinhamento
    # Eles são identificados pelos IDs:
    # 0 - Topo Esquerdo
    # 1 - Topo Direito
    # 2 - Baixo Esquerdo
    # 3 - Baixo Direito
    def aruco(self, input_path: str, output_path: str) -> NDArray[np.uint8]:
        img = cv2.imread(input_path)

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
        W, H = self.width, self.height

        dst = np.float32([ [0, 0], [W, 0], [W, H], [0, H] ]) # type: ignore

        # Matriz de transformação e warp
        M = cv2.getPerspectiveTransform(pts, dst) # type: ignore
        corrigida = cv2.warpPerspective(img, M, (W, H)) # type: ignore

        cv2.imwrite(output_path, corrigida)
        print("Imagem alinhada salva em", Color.green(output_path))
        return corrigida