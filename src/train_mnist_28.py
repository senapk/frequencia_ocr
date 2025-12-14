import numpy as np
import cv2
import gzip
import struct

BASE_URL = "https://github.com/golbin/TensorFlow-MNIST/tree/master/mnist/data/"
FILES = {
    "train_images": "raw/train-images-idx3-ubyte.gz",
    "train_labels": "raw/train-labels-idx1-ubyte.gz",
}

# def download(file: str):
#     if not os.path.exists(file):
#         urllib.request.urlretrieve(BASE_URL + file, file)

def load_images(path: str):
    with gzip.open(path, 'rb') as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, r * c)

def load_labels(path: str):
    with gzip.open(path, 'rb') as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

# download
# for f in FILES.values():
#     download(f)

# load
X = load_images(FILES["train_images"]).astype(np.float32)
y = load_labels(FILES["train_labels"]).astype(np.int32)

# (opcional) reduzir para testes rápidos
# X, y = X[:10000], y[:10000]

# train KNN
knn = cv2.ml.KNearest_create()
knn.train(X, cv2.ml.ROW_SAMPLE, y)

# save
knn.save("raw/mnist_knn_28x28.yml")
print("Salvo: raw/mnist_knn_28x28.yml")