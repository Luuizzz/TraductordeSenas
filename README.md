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


1. Instalación y Configuración
Clona el repositorio y prepara el entorno:


git clone <url-del-repo>
cd <nombre-directorio>
pip install -r requirements.txt
2. Recolección de Datos
Ejecuta el script para capturar las señas mediante video:


python collect_data.py
Asegúrate de grabar suficientes muestras para cada seña antes de pasar al siguiente paso.

3. Entrenamiento
Una vez recolectados los datos, inicia el entrenamiento del modelo:


python train_model.py
Al finalizar, el script generará automáticamente los siguientes archivos:

sign_model.keras

labels.json

4. Inferencia (Prueba en vivo)
Para ejecutar el reconocimiento en tiempo real con la cámara:


python inference.py

