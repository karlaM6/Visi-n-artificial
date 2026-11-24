import cv2
import numpy as np


def gaussiano(imagen, tamano=5):
    return cv2.GaussianBlur(imagen, (tamano, tamano), 0)


def sobel(gris):
    dx = cv2.Sobel(gris, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gris, cv2.CV_32F, 0, 1, ksize=3)
    magnitud = cv2.magnitude(dx, dy)
    return cv2.convertScaleAbs(magnitud)


def canny(gris):
    return cv2.Canny(gris, 100, 200)


def espectro_fourier(gris):
    transformada = np.fft.fft2(gris)
    desplazada = np.fft.fftshift(transformada)
    espectro = 20 * np.log(np.abs(desplazada) + 1)
    cv2.normalize(espectro, espectro, 0, 255, cv2.NORM_MINMAX)
    return espectro.astype(np.uint8)
