import cv2
import numpy as np


def ecualizar(gris):
    return cv2.equalizeHist(gris)


def clahe(gris, clip_limit=2.0, cuadricula=(8, 8)):
    transformador = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=cuadricula)
    return transformador.apply(gris)


def dibujar_histograma(gris):
    hist = cv2.calcHist([gris], [0], None, [256], [0, 256]).ravel()
    cv2.normalize(hist, hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    imagen = np.zeros((256, 256), dtype=np.uint8)
    for i in range(1, 256):
        cv2.line(
            imagen,
            (i - 1, 256 - int(hist[i - 1]) ),
            (i, 256 - int(hist[i]) ),
            255,
            1,
        )
    return imagen
