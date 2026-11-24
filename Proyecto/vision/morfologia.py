import cv2


def elemento(tipo="elipse", tamano=3):
    figura = {
        "rectangulo": cv2.MORPH_RECT,
        "cruz": cv2.MORPH_CROSS,
        "elipse": cv2.MORPH_ELLIPSE,
    }.get(tipo, cv2.MORPH_ELLIPSE)
    return cv2.getStructuringElement(figura, (tamano, tamano))


def erosion(imagen, kernel):
    return cv2.erode(imagen, kernel)


def dilatacion(imagen, kernel):
    return cv2.dilate(imagen, kernel)


def apertura(imagen, kernel):
    return cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)


def cierre(imagen, kernel):
    return cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)
