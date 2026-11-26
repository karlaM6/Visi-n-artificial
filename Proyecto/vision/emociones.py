import cv2


class DetectorEmociones:
    def __init__(self, cada_n_frames=5):
        self.cada_n_frames = cada_n_frames
        self.contador = 0
        self._emocion_por_rostro = {}
        self._deepface = None
        self._deepface_error = None
        ruta_cascada = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._clasificador = cv2.CascadeClassifier(ruta_cascada)

    def detectar_rostros(self, gris):
        rostros = self._clasificador.detectMultiScale(
            gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return [tuple(map(int, rostro)) for rostro in rostros]

    def _clasificar_emocion(self, cara):
        if self._deepface is None:
            from deepface import DeepFace
            self._deepface = DeepFace

        resultado = self._deepface.analyze(
            cara,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        if isinstance(resultado, list):
            resultado = resultado[0]
        emociones = resultado.get("emotion", {})
        etiqueta = max(emociones, key=emociones.get) if emociones else "Desconocida"
        traducciones = {
            "angry": "enojo",
            "disgust": "disgusto",
            "fear": "miedo",
            "happy": "feliz",
            "sad": "triste",
            "surprise": "sorpresa",
            "neutral": "neutral",
        }
        return traducciones.get(etiqueta.lower(), etiqueta)

    def _emocion(self, cara):
        try:
            return self._clasificar_emocion(cara)
        except ImportError:
            self._deepface_error = "DeepFace no esta instalado"
            return "No disponible"
        except Exception as error:
            self._deepface_error = str(error)
            print("Advertencia DeepFace:", error)
            return "No detectable"

    def etiquetas(self, frame, rostros):
        self.contador += 1
        if self.contador % self.cada_n_frames != 0:
            return [self._emocion_por_rostro.get(i, "Analizando...") for i in range(len(rostros))]

        try:
            for indice, rostro in enumerate(rostros):
                x, y, w, h = rostro
                cara = frame[y:y + h, x:x + w]
                if cara.size == 0:
                    continue
                self._emocion_por_rostro[indice] = self._clasificar_emocion(cara)
        except ImportError:
            self._deepface_error = "DeepFace no esta instalado"
            for indice in range(len(rostros)):
                self._emocion_por_rostro[indice] = "No disponible"
        except Exception as error:
            self._deepface_error = str(error)
            for indice in range(len(rostros)):
                self._emocion_por_rostro[indice] = "No detectable"
            print("Advertencia DeepFace:", error)

        return [self._emocion_por_rostro.get(i, "Analizando...") for i in range(len(rostros))]
