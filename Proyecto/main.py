"""Reconocimiento de emociones en tiempo real + pipeline de visión artificial.

Teclas:  q salir | p panel de procesamiento
         k caracterización | s captura de pantalla
"""
import argparse
import sys
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vision import camara, filtros, histogramas
from vision import segmentacion, caracterizacion

TAM = 160  # tamaño de cada celda del panel


def celda(img, titulo):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (TAM, TAM))
    cv2.putText(img, titulo, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return img


def construir_panel(roi_bgr):
    """Aplica el pipeline pertinente al rostro y devuelve (mosaico, caracteristicas)."""
    roi = cv2.resize(roi_bgr, (TAM, TAM))
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    eq = histogramas.ecualizar(gris)
    clahe = histogramas.clahe(gris)
    gaussiano = filtros.gaussiano(gris, 5)
    mascara, contorno = segmentacion.segmentar_piel(roi)
    seg = segmentacion.aplicar_mascara(roi, mascara)
    if contorno is not None:
        cv2.drawContours(seg, [contorno], -1, (0, 255, 0), 1)

    celdas = [
        celda(roi, "Original"), celda(gris, "Gris"), celda(eq, "Ecualizada"),
        celda(histogramas.dibujar_histograma(gris), "Histograma"),
        celda(clahe, "CLAHE"), celda(gaussiano, "Gaussiano"),
        celda(mascara, "Mascara piel"), celda(seg, "Segmentacion"),
    ]
    filas = [np.hstack(celdas[i:i + 4]) for i in range(0, len(celdas), 4)]
    return np.vstack(filas), caracterizacion.caracterizar(gris, mascara, contorno)


def texto_caracteristicas(c, frame, x0=10, y0=20):
    lineas = [f"area={c['area_px']:.0f}px  circ={c['circularidad']:.2f}  solidez={c['solidez']:.2f}",
              f"aspecto={c['relacion_aspecto']:.2f}  cobertura={c['cobertura']:.2f}",
              f"brillo={c['brillo_roi']:.0f}  contraste={c['contraste_roi']:.0f}  entropia={c['entropia_roi']:.2f}"]
    for i, l in enumerate(lineas):
        cv2.putText(frame, l, (x0, y0 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def ejecutar(args):
    from vision.emociones import DetectorEmociones   # import perezoso: DeepFace tarda en cargar

    try:
        cv2.namedWindow("Comprobacion de interfaz", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("Comprobacion de interfaz")
    except cv2.error as error:
        raise RuntimeError(
            "OpenCV fue instalado sin soporte de ventanas. Reinstala "
            "opencv-contrib-python 4.11.0.86 y no instales paquetes headless."
        ) from error

    cap = camara.abrir_camara(args.camara, 640, 480, 30)
    print("Propiedades de la cámara:", camara.propiedades(cap))
    detector = DetectorEmociones(cada_n_frames=5)
    panel_on, feats_on = True, True

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer el frame de la cámara.")
            break
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = detector.detectar_rostros(histogramas.ecualizar(gris))
        etiquetas = detector.etiquetas(frame, rostros)

        for (x, y, w, h), et in zip(rostros, etiquetas):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, et, (x, y - 10 if y - 10 > 10 else y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        if len(rostros) and (panel_on or feats_on):
            x, y, w, h = max(rostros, key=lambda r: r[2] * r[3])   # rostro más grande
            panel, carac = construir_panel(frame[y:y + h, x:x + w])
            if panel_on:
                cv2.imshow("Pipeline de procesamiento", panel)
            if feats_on and carac:
                texto_caracteristicas(carac, frame)

        cv2.imshow("Real-Time Face Emotion Recognition", frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        elif tecla == ord("p"):
            panel_on = not panel_on
            if not panel_on:
                cv2.destroyWindow("Pipeline de procesamiento")
        elif tecla == ord("k"):
            feats_on = not feats_on
        elif tecla == ord("s"):
            cv2.imwrite("captura.png", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camara", type=int, default=0)
    a = ap.parse_args()
    ejecutar(a)
