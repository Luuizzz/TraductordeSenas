"""
train_model.py - v4
Fixes:
  1. Augmentation solo sobre secuencias donde TODOS los frames tienen manos
  2. Clase sintética "ninguna" con secuencias de ceros + ruido mínimo
  3. El modelo aprende a callar cuando no hay señal
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
FEATURES   = 126
EPOCHS     = 100
BATCH_SIZE = 16

# Clase reservada para "sin seña" — no necesita carpeta en data/
NONE_CLASS        = "ninguna"
NONE_SAMPLES      = 200   # secuencias sintéticas de silencio

# ── Descubrimiento dinámico ───────────────────────────────────────────────────
def discover_classes(data_dir: str) -> list:
    classes = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if not os.path.isdir(path):
            continue
        npy_files = [f for f in os.listdir(path) if f.endswith(".npy")]
        if npy_files:
            classes.append(name)
        else:
            print(f"  [aviso] '{name}' sin .npy — ignorada")
    return classes

# ── Carga ─────────────────────────────────────────────────────────────────────
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
                print(f"    [skip] {sign}/{f} shape={seq.shape}")
        print(f"  {sign:15s}: {valid} válidas"
              + (f"  ({skipped} saltadas)" if skipped else ""))
    return np.array(X), np.array(y)

# ── Detectar si una secuencia tiene manos en todos los frames ─────────────────
def has_hands(seq: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Un frame sin manos = vector de ceros.
    Considera que hay manos si la norma media por frame supera el threshold.
    """
    norms = np.linalg.norm(seq, axis=1)   # (30,)
    return float(np.mean(norms)) > threshold

# ── Augmentation — SOLO sobre secuencias con manos ───────────────────────────
def augment_dataset(X: np.ndarray, y: np.ndarray, n: int = 3):
    X_aug, y_aug = [], []
    skipped = 0
    for seq, label in zip(X, y):
        if not has_hands(seq):
            skipped += 1
            continue
        for _ in range(n):
            noise = np.random.normal(0, 0.005, seq.shape)  # ruido más suave
            scale = np.random.uniform(0.97, 1.03)          # escala más conservadora
            X_aug.append(seq * scale + noise)
            y_aug.append(label)
    if skipped:
        print(f"  [augmentation] {skipped} secuencias sin manos excluidas del aug")
    return np.array(X_aug), np.array(y_aug)

# ── Clase "ninguna" sintética ─────────────────────────────────────────────────
def generate_none_class(n: int = NONE_SAMPLES) -> tuple:
    """
    Secuencias de ceros con ruido muy pequeño — representa ausencia de seña.
    También incluye variantes con una mano apareciendo y desapareciendo
    para que el modelo aprenda transiciones.
    """
    X_none = []

    # Puro silencio con ruido mínimo
    for _ in range(n // 2):
        seq = np.random.normal(0, 0.002, (SEQ_LENGTH, FEATURES))
        X_none.append(seq)

    # Mano que aparece brevemente (movimiento parcial)
    for _ in range(n // 2):
        seq = np.zeros((SEQ_LENGTH, FEATURES))
        start = np.random.randint(0, SEQ_LENGTH - 5)
        duration = np.random.randint(2, 6)
        for t in range(start, min(start + duration, SEQ_LENGTH)):
            seq[t] = np.random.normal(0.5, 0.1, FEATURES)  # landmarks aleatorios
        seq += np.random.normal(0, 0.002, seq.shape)
        X_none.append(seq)

    y_none = np.array([NONE_CLASS] * n)
    return np.array(X_none), y_none

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
        print("❌ No hay datos. Ejecuta collect_data.py primero.")
        return
    print(f"  Clases reales: {classes}")

    print("\nCargando dataset...")
    X, y_str = load_dataset(classes)
    print(f"  Total cargado: {len(X)}  shape={X.shape}")

    if len(X) == 0:
        print("❌ Ningún archivo cargado.")
        return

    # Augmentation solo sobre secuencias con manos detectadas
    print("\nAplicando augmentation (x3, solo secuencias con manos)...")
    X_aug, y_aug = augment_dataset(X, y_str, n=3)
    X     = np.concatenate([X, X_aug])
    y_str = np.concatenate([y_str, y_aug])
    print(f"  Total tras augmentation: {len(X)}")

    # Añadir clase "ninguna"
    print(f"\nGenerando clase '{NONE_CLASS}' ({NONE_SAMPLES} muestras sintéticas)...")
    X_none, y_none = generate_none_class(NONE_SAMPLES)
    X     = np.concatenate([X, X_none])
    y_str = np.concatenate([y_str, y_none])
    print(f"  Total final: {len(X)}")

    # Encoding
    le        = LabelEncoder()
    y         = le.fit_transform(y_str)
    label_map = {int(i): c for i, c in enumerate(le.classes_)}
    print(f"\nMapa de clases: {label_map}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)}  |  Val: {len(X_val)}")

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