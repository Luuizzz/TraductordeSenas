"""
train_model.py
Lee los .npy de data/, entrena un LSTM y guarda el modelo.

Requiere: pip install tensorflow scikit-learn matplotlib
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import json

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = "data"
MODEL_OUT   = "sign_model.keras"
LABELS_OUT  = "labels.json"
SEQ_LENGTH  = 30
FEATURES    = 225   # 63+63+99
EPOCHS      = 100
BATCH_SIZE  = 16

SIGNS = ["hola", "bien", "como_estas"]

# ── Cargar datos ──────────────────────────────────────────────────────────────
def load_dataset():
    X, y = [], []
    for sign in SIGNS:
        sign_dir = os.path.join(DATA_DIR, sign)
        if not os.path.isdir(sign_dir):
            print(f"⚠️  Carpeta no encontrada: {sign_dir}")
            continue
        files = sorted([f for f in os.listdir(sign_dir) if f.endswith(".npy")])
        print(f"  {sign}: {len(files)} muestras")
        for f in files:
            seq = np.load(os.path.join(sign_dir, f))
            if seq.shape == (SEQ_LENGTH, FEATURES):
                X.append(seq)
                y.append(sign)
            else:
                print(f"    [skip] shape inesperado: {seq.shape} en {f}")
    return np.array(X), np.array(y)

# ── Modelo ────────────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> tf.keras.Model:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, FEATURES)),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ── Entrenamiento ─────────────────────────────────────────────────────────────
def train():
    print("Cargando dataset...")
    X, y_str = load_dataset()
    print(f"  Total: {len(X)} muestras, shape X={X.shape}")

    if len(X) == 0:
        print("❌ No se encontraron datos. Ejecuta collect_data.py primero.")
        return

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    label_map = {i: c for i, c in enumerate(le.classes_)}
    print(f"  Clases: {label_map}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_model(num_classes=len(le.classes_))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5),
        ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor="val_accuracy"),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    # ── Métricas ──────────────────────────────────────────────────────────────
    y_pred = np.argmax(model.predict(X_val), axis=1)
    print("\n=== Reporte de clasificación ===")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_val, y_pred)
    print("Matriz de confusión:")
    print(cm)

    # ── Guardar labels ────────────────────────────────────────────────────────
    with open(LABELS_OUT, "w") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Modelo guardado: {MODEL_OUT}")
    print(f"✅ Labels guardados: {LABELS_OUT}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"], label="train")
    ax1.plot(history.history["val_accuracy"], label="val")
    ax1.set_title("Accuracy"); ax1.legend()
    ax2.plot(history.history["loss"], label="train")
    ax2.plot(history.history["val_loss"], label="val")
    ax2.set_title("Loss"); ax2.legend()
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=120)
    print("✅ Gráfico guardado: training_history.png")

if __name__ == "__main__":
    train()
