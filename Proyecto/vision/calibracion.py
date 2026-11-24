import os

import cv2
import numpy as np


def cargar_mapas(ruta):
    if not os.path.isfile(ruta):
        return None
    try:
        datos = np.load(ruta)
        return datos["mtx"], datos["dist"]
    except (OSError, KeyError):
        return None


def corregir_distorsion(frame, mapas):
    mtx, dist = mapas
    alto, ancho = frame.shape[:2]
    nueva_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (ancho, alto), 1)
    return cv2.undistort(frame, mtx, dist, None, nueva_mtx)


def calibrar(carpeta, patron, cuadro_mm, salida, mostrar=True):
    os.makedirs(carpeta, exist_ok=True)
    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    puntos_objeto = np.zeros((patron[0] * patron[1], 3), np.float32)
    puntos_objeto[:, :2] = np.mgrid[0:patron[0], 0:patron[1]].T.reshape(-1, 2)
    puntos_objeto *= cuadro_mm

    objetos, imagenes = [], []
    for nombre in sorted(os.listdir(carpeta)):
        ruta = os.path.join(carpeta, nombre)
        imagen = cv2.imread(ruta)
        if imagen is None:
            continue
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        encontrada, esquinas = cv2.findChessboardCorners(gris, tuple(patron), None)
        if encontrada:
            esquinas = cv2.cornerSubPix(gris, esquinas, (11, 11), (-1, -1), criterios)
            objetos.append(puntos_objeto)
            imagenes.append(esquinas)
            if mostrar:
                cv2.drawChessboardCorners(imagen, tuple(patron), esquinas, encontrada)
                cv2.imshow("Calibracion", imagen)
                cv2.waitKey(150)

    if len(objetos) < 3:
        if mostrar:
            cv2.destroyAllWindows()
        raise RuntimeError("Se necesitan al menos 3 imagenes con el patron")

    alto, ancho = cv2.imread(os.path.join(carpeta, sorted(os.listdir(carpeta))[0])).shape[:2]
    error, mtx, dist, _, _ = cv2.calibrateCamera(
        objetos, imagenes, (ancho, alto), None, None
    )
    np.savez(salida, mtx=mtx, dist=dist, error=np.array(error))
    print(f"Calibracion guardada en {salida}; error={error:.4f}")
    if mostrar:
        cv2.destroyAllWindows()
