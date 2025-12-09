import cv2
import numpy as np

# https://github.com/opencv/opencv/blob/4.x/samples/data/digits.png
def extract():
    img = cv2.imread("raw/digits.png", cv2.IMREAD_GRAYSCALE)

    cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]
    cells = np.array(cells)

    train = cells.reshape(-1, 400).astype(np.float32)

    k = np.arange(10)
    responses = np.repeat(k, 500)

    np.savetxt("raw/samples.data", train)
    np.savetxt("raw/responses.data", responses)

def train():
    # carregar seus arquivos
    samples = np.loadtxt("raw/samples.data", np.float32)
    responses = np.loadtxt("raw/responses.data", np.float32)

    # treinar modelo KNN
    knn = cv2.ml.KNearest_create()
    knn.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # salvar o modelo para uso posterior
    knn.save("raw/digits_knn.xml")

train()