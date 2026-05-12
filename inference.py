"""
inference.py - v3
Compatible con collect_data.py v3 y train_model.py v3.
- Solo manos (126 features)
- Dibujo manual de landmarks (sin solutions.drawing_utils)
- Ventana deslizante + suavizado por mayoría
- HUD con confianza y barra de progreso
"""

import cv2
import numpy as np
import json
import collections
import mediapipe as mp
import tensorflow as tf

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH_KERAS     = "sign_model.keras"
LABELS_PATH          = "labels.json"
HAND_TASK            = "models/hand_landmarker.task"
SEQ_LENGTH           = 30
FEATURES             = 126
CONFIDENCE_THRESHOLD = 0.80
SMOOTHING_WINDOW     = 5

CAM_WIDTH  = 1920
CAM_HEIGHT = 1080
CAM_FPS    = 30

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# ── Cargar modelo y labels ────────────────────────────────────────────────────
print("Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH_KERAS)

with open(LABELS_PATH, encoding="utf-8") as f:
    label_map = json.load(f)   # {"0": "bien", "1": "hola", ...}

print(f"Clases: {label_map}")

# ── MediaPipe ─────────────────────────────────────────────────────────────────
def build_hand_landmarker():
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_TASK),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)

# ── Dibujar malla ─────────────────────────────────────────────────────────────
def draw_hands(frame, hand_result):
    if not hand_result or not hand_result.hand_landmarks:
        return
    h, w = frame.shape[:2]
    for hand_landmarks in hand_result.hand_landmarks:
        for start_idx, end_idx in HAND_CONNECTIONS:
            x1 = int(hand_landmarks[start_idx].x * w)
            y1 = int(hand_landmarks[start_idx].y * h)
            x2 = int(hand_landmarks[end_idx].x * w)
            y2 = int(hand_landmarks[end_idx].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for lm in hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

# ── Extracción de features ────────────────────────────────────────────────────
def extract_features(hand_result) -> np.ndarray:
    rh = np.zeros(63)
    lh = np.zeros(63)
    if hand_result and hand_result.hand_landmarks:
        for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[i][0].category_name
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
            ).flatten()
            if handedness == "Right":
                rh = coords
            else:
                lh = coords
    return np.concatenate([rh, lh])

# ── HUD ───────────────────────────────────────────────────────────────────────
def draw_hud(frame, hand_result, sequence, current_sign, confidence):
    h, w = frame.shape[:2]
    n_hands = len(hand_result.hand_landmarks) if hand_result and hand_result.hand_landmarks else 0

    # Banda superior
    cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)

    if current_sign:
        label_text = f"{current_sign.upper()}  ({confidence*100:.0f}%)"
        cv2.putText(frame, label_text, (20, 55),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 255, 120), 2)
    else:
        cv2.putText(frame, "...", (20, 55),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, (120, 120, 120), 2)

    # Indicador manos
    dot_color = (0, 230, 80) if n_hands > 0 else (0, 50, 220)
    cv2.circle(frame, (w - 60, 38), 18, dot_color, -1)
    cv2.putText(frame, str(n_hands), (w - 69, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "manos", (w - 88, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Barra de ventana (frames acumulados)
    bar_w = int((len(sequence) / SEQ_LENGTH) * w)
    cv2.rectangle(frame, (0, h - 8), (w, h), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, h - 8), (bar_w, h), (0, 200, 255), -1)

    # Atajo salir
    cv2.putText(frame, "Q = salir", (20, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

# ── Main ─────────────────────────────────────────────────────────────────────
def run():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Cámara: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

    hand_det           = build_hand_landmarker()
    sequence           = collections.deque(maxlen=SEQ_LENGTH)
    predictions_buffer = collections.deque(maxlen=SMOOTHING_WINDOW)
    timestamp_ms       = 0
    current_sign       = ""
    confidence_val     = 0.0

    print("Inference iniciada. Presiona Q para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms += 33

        hand_result = hand_det.detect_for_video(mp_image, timestamp_ms)

        draw_hands(frame, hand_result)

        features = extract_features(hand_result)
        sequence.append(features)

        # Predicción cuando la ventana está llena
        if len(sequence) == SEQ_LENGTH:
            X     = np.expand_dims(np.array(sequence), axis=0)  # (1, 30, 126)
            probs = model.predict(X, verbose=0)[0]
            pred_idx  = int(np.argmax(probs))
            pred_conf = float(probs[pred_idx])

            predictions_buffer.append(pred_idx if pred_conf >= CONFIDENCE_THRESHOLD else -1)

            # Mayoría en buffer de suavizado
            from collections import Counter
            most_common, count = Counter(predictions_buffer).most_common(1)[0]
            if most_common != -1 and count > SMOOTHING_WINDOW // 2:
                current_sign   = label_map[str(most_common)]
                confidence_val = pred_conf
            else:
                current_sign = ""

        draw_hud(frame, hand_result, sequence, current_sign, confidence_val)
        cv2.imshow("Sign Language - Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hand_det.close()

if __name__ == "__main__":
    run()