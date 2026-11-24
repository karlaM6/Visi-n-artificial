import cv2


def segmentar_piel(bgr):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    mascara = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contorno = max(contornos, key=cv2.contourArea) if contornos else None
    if contorno is not None:
        mascara_plana = mascara.copy()
        mascara = mascara * 0
        cv2.drawContours(mascara, [contorno], -1, 255, cv2.FILLED)
    return mascara, contorno


def aplicar_mascara(bgr, mascara):
    return cv2.bitwise_and(bgr, bgr, mask=mascara)
