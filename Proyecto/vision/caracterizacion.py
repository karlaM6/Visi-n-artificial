import cv2
import numpy as np


def caracterizar(gris, mascara, contorno):
    if contorno is None:
        return None

    area = float(cv2.contourArea(contorno))
    perimetro = float(cv2.arcLength(contorno, True))
    _, _, ancho, alto = cv2.boundingRect(contorno)
    relacion_aspecto = ancho / float(alto) if alto else 0.0
    circularidad = (4 * np.pi * area / (perimetro * perimetro)) if perimetro else 0.0
    casco = cv2.convexHull(contorno)
    area_casco = cv2.contourArea(casco)
    solidez = area / float(area_casco) if area_casco else 0.0
    cobertura = area / float(gris.shape[0] * gris.shape[1])

    pixeles = gris[mascara > 0]
    brillo = float(np.mean(pixeles)) if pixeles.size else float(np.mean(gris))
    contraste = float(np.std(pixeles)) if pixeles.size else float(np.std(gris))
    entropia = 0.0
    hist = cv2.calcHist([gris], [0], mascara, [256], [0, 256]).ravel()
    total = hist.sum()
    if total:
        probabilidades = hist[hist > 0] / total
        entropia = float(-np.sum(probabilidades * np.log2(probabilidades)))

    return {
        "area_px": area,
        "perimetro_px": perimetro,
        "circularidad": circularidad,
        "solidez": solidez,
        "relacion_aspecto": relacion_aspecto,
        "cobertura": cobertura,
        "brillo_roi": brillo,
        "contraste_roi": contraste,
        "entropia_roi": entropia,
    }
