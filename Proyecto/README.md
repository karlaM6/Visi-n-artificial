# Reconocimiento de Emociones en Tiempo Real — Visión Artificial

Karla Martinez Visión Artificial · Pontificia Universidad Javeriana

Aplicación que detecta rostros con la webcam, clasifica la emoción dominante (en español) y
muestra en vivo el pipeline pertinente de visión artificial aplicado al rostro:
filtros, histogramas, segmentación y caracterización.

## Ramificación (Git)

| Rama | Propósito |
|------|-----------|
| `main` | Versión estable y entregable |
| `develop` | Integración de funcionalidades |
| `feature/filtros-histogramas` | Filtros espaciales, Fourier e histogramas |
| `feature/umbral-morfologia` | Umbralización/binarización y morfología (erosión, dilatación) |
| `feature/segmentacion-caracterizacion` | Segmentación de piel y descriptores |
| `feature/emociones` | Haar Cascade + DeepFace |

Flujo: `feature/*` → PR a `develop` → merge a `main` al entregar.

## Descripción de la arquitectura

```
main.py                  Orquestador: bucle de video, teclas, ventanas
└── vision/
    ├── camara.py            Modelo de cámara: VideoCapture, propiedades (ancho, alto, fps)
    ├── filtros.py           Caja, gaussiano, mediana, Sobel, Laplaciano, Canny, nitidez, afines (rotar/reflejar), espectro de Fourier
    ├── histogramas.py       Histograma, ecualización, CLAHE, brillo/contraste/entropía
    ├── umbralizacion.py     Global, Otsu, adaptativa (media/gaussiana)
    ├── morfologia.py        Erosión, dilatación, apertura, cierre, gradiente (elemento estructurante)
    ├── segmentacion.py      Piel en YCrCb + morfología + contorno mayor
    ├── caracterizacion.py   Área, perímetro, circularidad, solidez, aspecto, centroide, momentos de Hu, intensidad
    └── emociones.py         Haar Cascade (detección) + DeepFace (emoción)
```

**Flujo de datos por frame**

```
Cámara → gris → ecualización → Haar Cascade → rostros
   rostros → DeepFace → emoción (cada 5 frames)
   rostro más grande → filtros / umbral / morfología / segmentación → caracterización
```

## Instalación y uso

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
pip install deepface --no-deps
python main.py
```

Teclas: `q` salir · `p` panel de procesamiento · `k` caracterización · `s` captura.

## Referencias
Material del curso de Visión Artificial (cámaras, filtros, histogramas, umbralización, morfología, segmentación); documentación de OpenCV y DeepFace.
