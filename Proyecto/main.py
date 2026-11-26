"""Reconocimiento de emociones en tiempo real + visión artificial que INFLUYE en la decisión.

Qué hace el procesamiento (no es solo decorativo):
  1. Calidad de iluminación (histograma): si hay poca luz / sobreexposición / bajo contraste,
     NO se clasifica la emoción (evita resultados poco confiables).
  2. Validación con segmentación + caracterización: si la región de piel del "rostro" es
     pequeña o irregular, se descarta como posible falso positivo del Haar Cascade.
  3. Comparación A/B: emoción con el rostro original vs. preprocesado (CLAHE + gaussiano).

Teclas: q salir | p panel | k características | v validación on/off
        a comparar original vs. preprocesado | s captura
"""
import argparse
import sys
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vision import camara, filtros, histogramas
from vision import segmentacion, caracterizacion

TAM = 160

BRILLO_MIN, BRILLO_MAX, CONTRASTE_MIN = 60, 200, 20
COBERTURA_MIN, SOLIDEZ_MIN = 0.10, 0.60

VERDE, NARANJA, ROJO, AMARILLO = (0, 255, 0), (0, 165, 255), (0, 0, 255), (0, 255, 255)


def analizar_rostro(roi_bgr, validar=True):
    """Devuelve un dict con las etapas del procesamiento y el estado del rostro:
    'ok' (se clasifica), 'calidad' (iluminación mala) o 'descartado' (falso positivo)."""
    roi = cv2.resize(roi_bgr, (TAM, TAM))
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mascara, contorno = segmentacion.segmentar_piel(roi)
    carac = caracterizacion.caracterizar(gris, mascara, contorno)
    stats = histogramas.estadisticas(gris)

    estado, motivo = "ok", ""
    if stats["brillo"] < BRILLO_MIN:
        estado, motivo = "calidad", "Poca luz"
    elif stats["brillo"] > BRILLO_MAX:
        estado, motivo = "calidad", "Sobreexpuesto"
    elif stats["contraste"] < CONTRASTE_MIN:
        estado, motivo = "calidad", "Bajo contraste"
    elif validar and (carac is None or carac["cobertura"] < COBERTURA_MIN
                      or carac["solidez"] < SOLIDEZ_MIN):
        estado, motivo = "descartado", "Posible falso positivo"

    return {"roi": roi, "gris": gris, "mascara": mascara, "contorno": contorno,
            "carac": carac, "estado": estado, "motivo": motivo}


def preprocesar_para_emocion(roi_bgr):
    """CLAHE sobre el canal L (LAB) + suavizado gaussiano leve."""
    l, a, b = cv2.split(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB))
    lab = cv2.merge((histogramas.clahe(l), a, b))
    return filtros.gaussiano(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), 3)


def celda(img, titulo):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (TAM, TAM))
    cv2.putText(img, titulo, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, AMARILLO, 1, cv2.LINE_AA)
    return img


def construir_panel(a):
    roi, gris = a["roi"], a["gris"]
    seg = segmentacion.aplicar_mascara(roi, a["mascara"])
    if a["contorno"] is not None:
        cv2.drawContours(seg, [a["contorno"]], -1, VERDE, 1)
    celdas = [
        celda(roi, "Original"), celda(gris, "Gris"), celda(histogramas.ecualizar(gris), "Ecualizada"),
        celda(histogramas.dibujar_histograma(gris), "Histograma"),
        celda(histogramas.clahe(gris), "CLAHE"), celda(filtros.gaussiano(gris, 5), "Gaussiano"),
        celda(a["mascara"], "Mascara piel"), celda(seg, "Segmentacion"),
    ]
    return np.vstack([np.hstack(celdas[i:i + 4]) for i in range(0, len(celdas), 4)])


def texto(frame, lineas, x0, y0, color=AMARILLO, escala=0.5):
    for i, l in enumerate(lineas):
        cv2.putText(frame, l, (x0, y0 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, escala, color, 1, cv2.LINE_AA)


def texto_caracteristicas(c, frame):
    texto(frame, [f"area={c['area_px']:.0f}px  circ={c['circularidad']:.2f}  solidez={c['solidez']:.2f}",
                  f"aspecto={c['relacion_aspecto']:.2f}  cobertura={c['cobertura']:.2f}",
                  f"brillo={c['brillo_roi']:.0f}  contraste={c['contraste_roi']:.0f}  "
                  f"entropia={c['entropia_roi']:.2f}"], 10, 20)


class Estado:
    def __init__(self, cada_n=5):
        self.cada_n, self.n_frame, self.n_rostros = cada_n, 0, -1
        self.cache = {}
        self.coinciden, self.total = 0, 0


def procesar_frame(frame, detector, emocion_fn, est, validar=True, comparar=False):
    """Dibuja sobre `frame` y devuelve el análisis del rostro más grande (o None)."""
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = detector.detectar_rostros(histogramas.ecualizar(gris))
    est.n_frame += 1
    recalcular = est.n_frame % est.cada_n == 0 or len(rostros) != est.n_rostros
    est.n_rostros = len(rostros)

    analisis_mayor, area_mayor = None, 0
    for i, (x, y, w, h) in enumerate(rostros):
        roi = frame[y:y + h, x:x + w]
        a = analizar_rostro(roi, validar)
        if w * h > area_mayor:
            analisis_mayor, area_mayor = a, w * h

        if a["estado"] == "ok":
            if recalcular or i not in est.cache:
                orig = emocion_fn(roi)
                proc = emocion_fn(preprocesar_para_emocion(roi)) if comparar else None
                est.cache[i] = (orig, proc)
                if comparar:
                    est.total += 1
                    est.coinciden += int(orig == proc)
            orig, proc = est.cache[i]
            color = VERDE
            etiqueta = f"O:{orig}  P:{proc}" if comparar and proc else orig
        else:
            est.cache.pop(i, None)
            color = NARANJA if a["estado"] == "calidad" else ROJO
            etiqueta = a["motivo"]

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, etiqueta, (x, y - 10 if y - 10 > 10 else y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 if not comparar else 0.6, color, 2, cv2.LINE_AA)
    return analisis_mayor


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
    detector = DetectorEmociones()
    est = Estado(cada_n=5)
    panel_on, feats_on, validar, comparar = True, True, True, False

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer el frame de la cámara.")
            break

        a = procesar_frame(frame, detector, detector._emocion, est, validar, comparar)
        if a is not None:
            if panel_on:
                cv2.imshow("Pipeline de procesamiento", construir_panel(a))
            if feats_on and a["carac"]:
                texto_caracteristicas(a["carac"], frame)

        estado = [f"validacion: {'ON' if validar else 'OFF'}",
                  f"comparar A/B: {'ON' if comparar else 'OFF'}"]
        if comparar and est.total:
            estado.append(f"coinciden: {100 * est.coinciden / est.total:.0f}% ({est.total} muestras)")
        texto(frame, estado, 10, frame.shape[0] - 10 - 18 * (len(estado) - 1))

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
        elif tecla == ord("v"):
            validar = not validar
        elif tecla == ord("a"):
            comparar = not comparar
            est.coinciden = est.total = 0
            est.cache.clear()
        elif tecla == ord("s"):
            cv2.imwrite("captura.png", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camara", type=int, default=0)
    ejecutar(ap.parse_args())