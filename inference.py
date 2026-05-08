"""
inference.py
Reconocimiento en tiempo real usando el modelo entrenado.
Mantiene una ventana deslizante de SEQ_LENGTH frames.
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
MODEL_PATH_KERAS = "sign_model.keras"
LABELS_PATH      = "labels.json"
HAND_TASK        = "models/hand_landmarker.task"
POSE_TASK        = "models/pose_landmarker_lite.task"
SEQ_LENGTH       = 30
FEATURES         = 225
CONFIDENCE_THRESHOLD = 0.80
SMOOTHING_WINDOW     = 5   # mayoría en últimas N predicciones

# ── Cargar modelo y labels ────────────────────────────────────────────────────
model = tf.keras.models.load_model(MODEL_PATH_KERAS)
with open(LABELS_PATH) as f:
    label_map = json.load(f)   # {"0": "bien", "1": "como_estas", "2": "hola"}

# ── Inicializar MediaPipe ─────────────────────────────────────────────────────
def build_hand_landmarker():
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_TASK),
        num_hands=2,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)

def build_pose_landmarker():
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_TASK),
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)

def extract_features(hand_result, pose_result) -> np.ndarray:
    rh = np.zeros(63)
    lh = np.zeros(63)
    pose = np.zeros(99)
    if hand_result.hand_landmarks:
        for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[i][0].category_name
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()
            if handedness == "Right":
                rh = coords
            else:
                lh = coords
    if pose_result.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks[0]]).flatten()
    return np.concatenate([rh, lh, pose])

# ── Inference loop ────────────────────────────────────────────────────────────
def run():
    cap = cv2.VideoCapture(0)
    hand_det = build_hand_landmarker()
    pose_det = build_pose_landmarker()

    sequence = collections.deque(maxlen=SEQ_LENGTH)
    predictions_buffer = collections.deque(maxlen=SMOOTHING_WINDOW)
    timestamp_ms = 0
    current_sign = ""
    confidence_val = 0.0

    print("Inference iniciada. Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms += 33

        h_res = hand_det.detect_for_video(mp_image, timestamp_ms)
        p_res = pose_det.detect_for_video(mp_image, timestamp_ms)
        features = extract_features(h_res, p_res)
        sequence.append(features)

        if len(sequence) == SEQ_LENGTH:
            X = np.expand_dims(np.array(sequence), axis=0)  # (1, 30, 225)
            probs = model.predict(X, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            pred_conf = float(probs[pred_idx])

            if pred_conf >= CONFIDENCE_THRESHOLD:
                predictions_buffer.append(pred_idx)
            else:
                predictions_buffer.append(-1)

            # Mayoría en el buffer de suavizado
            if predictions_buffer:
                from collections import Counter
                most_common, count = Counter(predictions_buffer).most_common(1)[0]
                if most_common != -1 and count > SMOOTHING_WINDOW // 2:
                    current_sign = label_map[str(most_common)]
                    confidence_val = pred_conf
                else:
                    current_sign = ""

        # ── HUD ──────────────────────────────────────────────────────────────
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)

        if current_sign:
            text = f"{current_sign}  ({confidence_val*100:.0f}%)"
            cv2.putText(frame, text, (10, h - 25),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 120), 2)
        else:
            cv2.putText(frame, "...", (10, h - 25),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, (120, 120, 120), 2)

        # Barra de frames capturados
        bar_w = int((len(sequence) / SEQ_LENGTH) * w)
        cv2.rectangle(frame, (0, h - 5), (bar_w, h), (0, 200, 255), -1)

        cv2.imshow("Sign Language Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hand_det.close()
    pose_det.close()

if __name__ == "__main__":
    run()
