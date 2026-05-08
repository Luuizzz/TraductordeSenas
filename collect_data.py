"""
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
"""

import cv2
import numpy as np
import os
import mediapipe as mp

# ── MediaPipe Tasks API (>= 0.10 / 0.35) ─────────────────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Configuración ─────────────────────────────────────────────────────────────
SIGNS       = ["hola", "bien", "como_estas"]
DATA_DIR    = "data"
NUM_SAMPLES = 30      # muestras por seña
SEQ_LENGTH  = 30      # frames por muestra
MODEL_PATH  = "models/hand_landmarker.task"   # descarga abajo
POSE_PATH   = "models/pose_landmarker_lite.task"

# Descarga de modelos si no existen
HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
POSE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

def download_model(url: str, path: str):
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Descargando {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)
        print("  Listo.")

# ── Inicializar detectors ─────────────────────────────────────────────────────
def build_hand_landmarker():
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)

def build_pose_landmarker():
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_PATH),
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)

# ── Extracción de features ────────────────────────────────────────────────────
def extract_features(hand_result, pose_result) -> np.ndarray:
    """Devuelve vector (225,): 21*3 mano_d + 21*3 mano_i + 33*3 pose."""
    rh = np.zeros(63)
    lh = np.zeros(63)
    pose = np.zeros(99)

    # Manos
    if hand_result.hand_landmarks:
        for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[i][0].category_name  # "Left" / "Right"
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()
            if handedness == "Right":
                rh = coords
            else:
                lh = coords

    # Pose
    if pose_result.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks[0]]).flatten()

    return np.concatenate([rh, lh, pose])

# ── Recolección ───────────────────────────────────────────────────────────────
def collect():
    download_model(HAND_URL, MODEL_PATH)
    download_model(POSE_URL, POSE_PATH)

    for sign in SIGNS:
        os.makedirs(os.path.join(DATA_DIR, sign), exist_ok=True)

    cap = cv2.VideoCapture(0)
    hand_det = build_hand_landmarker()
    pose_det = build_pose_landmarker()
    timestamp_ms = 0

    for sign in SIGNS:
        for sample_idx in range(NUM_SAMPLES):
            save_path = os.path.join(DATA_DIR, sign, f"{sample_idx}.npy")
            if os.path.exists(save_path):
                print(f"  [skip] {sign}/{sample_idx}.npy ya existe")
                continue

            print(f"\n[LISTO] Seña: '{sign}'  Muestra: {sample_idx+1}/{NUM_SAMPLES}")
            print("  Presiona ESPACIO para iniciar la captura...")

            # Esperar ESPACIO
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                overlay = frame.copy()
                cv2.putText(overlay, f"{sign} [{sample_idx+1}/{NUM_SAMPLES}]",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(overlay, "ESPACIO = iniciar",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow("Captura", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    break
                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # Capturar SEQ_LENGTH frames
            sequence = []
            for frame_idx in range(SEQ_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms += 33  # ~30 fps

                h_res = hand_det.detect_for_video(mp_image, timestamp_ms)
                p_res = pose_det.detect_for_video(mp_image, timestamp_ms)

                features = extract_features(h_res, p_res)
                sequence.append(features)

                cv2.putText(frame, f"Grabando... {frame_idx+1}/{SEQ_LENGTH}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Captura", frame)
                cv2.waitKey(1)

            np.save(save_path, np.array(sequence))
            print(f"  Guardado: {save_path}  shape={np.array(sequence).shape}")

    cap.release()
    cv2.destroyAllWindows()
    hand_det.close()
    pose_det.close()
    print("\n✅ Recolección completa.")

if __name__ == "__main__":
    collect()
