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


Orden de ejecucion
Clonar repo
moverse al directorio de Instalacion
Ejecutar pip install -r requirements.txt
Una vez terminada la instalacion Hacer 
python collect_data.py
Llenar los videos de las senas
Una vez llenado, ejecutar
python train_model.py
al ejecutarlo se iniciara el entrenamiento y generara  → sign_model.keras + labels.json
Una vez terminado el entrenamiento se puede ejecutar inference.py, alli se abrira la ventana con analisis en tiempo real


