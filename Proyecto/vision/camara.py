import cv2


def abrir_camara(indice=0, ancho=640, alto=480, fps=30):
    cap = cv2.VideoCapture(indice)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la camara")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def propiedades(cap):
    return {
        "ancho": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "alto": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
    }
