collect_data.py
Captura secuencias de landmarks para cada seña y las guarda como .npy.

Estructura esperada/generada:
  data/
    hola/       -> 30 archivos: 0.npy ... 29.npy
    bien/
    como_estas/

Cada .npy tiene shape (30, 225):
  - 30 frames por muestra
  - 21 puntos mano derecha * 3 + 21 mano izquierda * 3 + 33 pose * 3 = 225

  train_model.py
Lee los .npy de data/, entrena un LSTM y guarda el modelo.

inference.py
Reconocimiento en tiempo real usando el modelo entrenado.
Mantiene una ventana deslizante de SEQ_LENGTH frames.


# Sign Language Recognition

Guía de instalación y uso del sistema.

## Pasos de Instalación

1. Clonar el repositorio

2. Instalar dependencias:
   pip install -r requirements.txt

## Uso del Sistema

1. Recolección de datos:
   Ejecuta el script para grabar los videos de las señas:
   python collect_data.py

   como ya esta entrenado solo ejecutar para instalar mediapipe

2. Entrenamiento:
   Una vez recolectados los videos, inicia el entrenamiento:
   python train_model.py
   (Esto generará los archivos sign_model.keras y labels.json)
   (ahora mismo con el clone queda listo para no entrenarlo)

4. Inferencia:
   Para probar el modelo en tiempo real con la cámara:
   python inference.py
