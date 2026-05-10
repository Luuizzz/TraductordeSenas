"""
train_model.py - v3
Compatible con collect_data.py v3 (manos únicamente, 126 features).
Descubre clases dinámicamente desde data/ — sin lista hardcodeada.
Lee todos los .npy de cada carpeta sin límite de archivos.

Estructura esperada:
  data/
    hola/     *.npy   shape (30, 126)
    bien/     *.npy
    ...       (cualquier carpeta nueva se incluye automáticamente)
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
DATA_DIR   = "data"
MODEL_OUT  = "sign_model.keras"
LABELS_OUT = "labels.json"
SEQ_LENGTH = 30
FEATURES   = 126   # 63 mano_d + 63 mano_i  (sin pose, sin cara)
EPOCHS     = 100
BATCH_SIZE = 16

# ── Descubrimiento dinámico de clases ─────────────────────────────────────────
def discover_classes(data_dir: str) -> list:
    """Subcarpetas de data/ que tengan al menos un .npy válido."""
    classes = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if not os.path.isdir(path):
            continue
        npy_files = [f for f in os.listdir(path) if f.endswith(".npy")]
        if npy_files:
            classes.append(name)
        else:
            print(f"  [aviso] '{name}' existe pero no tiene .npy — ignorada")
    return classes

# ── Cargar dataset ────────────────────────────────────────────────────────────
def load_dataset(classes: list):
    X, y = [], []
    for sign in classes:
        sign_dir = os.path.join(DATA_DIR, sign)
        files    = [f for f in os.listdir(sign_dir) if f.endswith(".npy")]
        valid, skipped = 0, 0
        for f in files:
            seq = np.load(os.path.join(sign_dir, f))
            if seq.shape == (SEQ_LENGTH, FEATURES):
                X.append(seq)
                y.append(sign)
                valid += 1
            else:
                skipped += 1
                print(f"    [skip] {sign}/{f} shape={seq.shape} — esperado ({SEQ_LENGTH},{FEATURES})")
        print(f"  {sign:15s}: {valid} válidas"
              + (f"  ({skipped} saltadas)" if skipped else ""))
    return np.array(X), np.array(y)

# ── Augmentation ──────────────────────────────────────────────────────────────
def augment_dataset(X: np.ndarray, y: np.ndarray, n: int = 3):
    """Genera n variantes por muestra con ruido y escala leve."""
    X_aug, y_aug = [], []
    for seq, label in zip(X, y):
        for _ in range(n):
            noise = np.random.normal(0, 0.01, seq.shape)
            scale = np.random.uniform(0.95, 1.05)
            X_aug.append(seq * scale + noise)
            y_aug.append(label)
    return np.array(X_aug), np.array(y_aug)

# ── Modelo ────────────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> tf.keras.Model:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, FEATURES)),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(32, return_sequences=False),
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
    print(f"\nDescubriendo clases en '{DATA_DIR}/'...")
    classes = discover_classes(DATA_DIR)
    if not classes:
        print("❌ No se encontraron carpetas con datos. Ejecuta collect_data.py primero.")
        return
    print(f"  Clases: {classes}\n")

    print("Cargando dataset...")
    X, y_str = load_dataset(classes)
    print(f"\n  Total cargado: {len(X)} muestras  shape={X.shape}")

    if len(X) == 0:
        print("❌ Ningún archivo cargado.")
        return

    # Augmentation
    print("\nAplicando augmentation x3...")
    X_aug, y_aug = augment_dataset(X, y_str, n=3)
    X     = np.concatenate([X, X_aug])
    y_str = np.concatenate([y_str, y_aug])
    print(f"  Total tras augmentation: {len(X)} muestras")

    # Encoding
    le        = LabelEncoder()
    y         = le.fit_transform(y_str)
    label_map = {int(i): c for i, c in enumerate(le.classes_)}
    print(f"\nMapa de clases: {label_map}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)}  |  Val: {len(X_val)}")

    model = build_model(num_classes=len(classes))
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

    # Métricas
    y_pred = np.argmax(model.predict(X_val), axis=1)
    print("\n=== Reporte de clasificación ===")
    print(classification_report(y_val, y_pred, target_names=le.classes_))
    print("Matriz de confusión:")
    print(confusion_matrix(y_val, y_pred))

    # Guardar labels
    with open(LABELS_OUT, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Modelo: {MODEL_OUT}")
    print(f"✅ Labels: {LABELS_OUT}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"],     label="train")
    ax1.plot(history.history["val_accuracy"], label="val")
    ax1.set_title("Accuracy"); ax1.legend()
    ax2.plot(history.history["loss"],     label="train")
    ax2.plot(history.history["val_loss"], label="val")
    ax2.set_title("Loss"); ax2.legend()
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=120)
    print("✅ Gráfico: training_history.png")

if __name__ == "__main__":
    train()
