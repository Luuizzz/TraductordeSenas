"""
collect_data.py - v3
Solo manos (126 features). Sin pose ni cara.
Cámara en 1280x720 o 1920x1080 según soporte del hardware.
Malla de landmarks dibujada en tiempo real.
"""

import cv2
import numpy as np
import os
import mediapipe as mp
import urllib.request

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ── Config ────────────────────────────────────────────────────────────────────
SIGNS = [
    "bien", "hola", "como_estas", "si", "no",
    "gracias", "yo", "tu", "comer", "amigo",
]

HINTS = {
    "bien":       "PULGAR ARRIBA  👍",
    "hola":       "MANO ABIERTA MOVIENDOSE  👋",
    "como_estas": "DOS MANOS AL FRENTE CON MOVIMIENTO  🤲",
    "si":         "PUNO ARRIBA/ABAJO  ✊",
    "no":         "INDICE Y MEDIO CERRANDOSE  ✌️",
    "gracias":    "MANO DESDE LA BOCA HACIA AFUERA  🙏",
    "yo":         "INDICE AL PECHO  👆",
    "tu":         "INDICE AL FRENTE  👉",
    "comer":      "DEDOS JUNTOS HACIA LA BOCA  🤌",
    "amigo":      "MANOS ENTRELAZADAS / PUNOS JUNTOS  🤝",
}

DATA_DIR    = "data"
NUM_SAMPLES = 80
SEQ_LENGTH  = 30
FEATURES    = 126    # 63 mano_d + 63 mano_i

# Resolución deseada — OpenCV la acepta si la cámara la soporta
CAM_WIDTH  = 1920
CAM_HEIGHT = 1080
CAM_FPS    = 30

HAND_TASK = "models/hand_landmarker.task"
HAND_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# ── Descarga modelo ───────────────────────────────────────────────────────────
def download_model(url: str, path: str):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Descargando {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)
        print("  Listo.")

# ── Cámara ────────────────────────────────────────────────────────────────────
def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # CAP_DSHOW = menos latencia en Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Cámara iniciada: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")
    return cap

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

    # conexiones de la mano
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17)
    ]

    for hand_landmarks in hand_result.hand_landmarks:

        # Dibujar conexiones
        for start_idx, end_idx in HAND_CONNECTIONS:

            x1 = int(hand_landmarks[start_idx].x * w)
            y1 = int(hand_landmarks[start_idx].y * h)

            x2 = int(hand_landmarks[end_idx].x * w)
            y2 = int(hand_landmarks[end_idx].y * h)

            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Dibujar puntos
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
def draw_hud(frame, sign, hint, sample_idx, total, state, hand_result, frames_done=0):
    h, w = frame.shape[:2]
    n_hands = len(hand_result.hand_landmarks) if hand_result and hand_result.hand_landmarks else 0

    # Banda superior
    cv2.rectangle(frame, (0, 0), (w, 110), (20, 20, 20), -1)

    cv2.putText(frame, f"Sena: {sign.upper()}  [{sample_idx+1}/{total}]",
                (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
    cv2.putText(frame, hint,
                (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 0), 2)

    # Indicador manos — círculo + texto
    dot_color = (0, 230, 80) if n_hands > 0 else (0, 50, 220)
    cv2.circle(frame, (w - 60, 50), 20, dot_color, -1)
    cv2.putText(frame, str(n_hands), (w - 68, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, "manos", (w - 90, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Banda inferior
    cv2.rectangle(frame, (0, h - 70), (w, h), (20, 20, 20), -1)

    if state == "wait":
        cv2.putText(frame, "ESPACIO = grabar  |  S = saltar sena  |  Q = salir",
                    (20, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    else:
        cv2.putText(frame, f"Grabando: {frames_done}/{SEQ_LENGTH}",
                    (20, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 80, 255), 2)
        bar_w = int((frames_done / SEQ_LENGTH) * w)
        cv2.rectangle(frame, (0, h - 10), (bar_w, h), (0, 200, 255), -1)

# ── Recolección ───────────────────────────────────────────────────────────────
def collect():
    download_model(HAND_URL, HAND_TASK)
    for sign in SIGNS:
        os.makedirs(os.path.join(DATA_DIR, sign), exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Cámara: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

    hand_det     = build_hand_landmarker()
    timestamp_ms = 0
    hand_result  = None

    for sign in SIGNS:
        hint      = HINTS.get(sign, "")
        sign_dir  = os.path.join(DATA_DIR, sign)
        existing  = [f for f in os.listdir(sign_dir) if f.endswith(".npy")]
        start_idx = len(existing)

        if start_idx >= NUM_SAMPLES:
            print(f"[skip] '{sign}' ya tiene {start_idx} muestras.")
            continue

        print(f"\n► '{sign}'  ({start_idx} existentes, faltan {NUM_SAMPLES - start_idx})")
        skip_sign  = False
        sample_idx = start_idx

        while sample_idx < NUM_SAMPLES and not skip_sign:
            save_path = os.path.join(sign_dir, f"{sample_idx}.npy")

            # ── Esperar ESPACIO ───────────────────────────────────────────────
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms += 33
                hand_result = hand_det.detect_for_video(mp_image, timestamp_ms)

                draw_hands(frame, hand_result)
                draw_hud(frame, sign, hint, sample_idx, NUM_SAMPLES, "wait", hand_result)
                cv2.imshow("Recoleccion", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    break
                if key == ord("s"):
                    skip_sign = True
                    break
                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    hand_det.close()
                    print("\nInterrumpido.")
                    return

            if skip_sign:
                break

            # ── Grabar ────────────────────────────────────────────────────────
            sequence = []
            for frame_idx in range(SEQ_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    break

                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms += 33
                hand_result = hand_det.detect_for_video(mp_image, timestamp_ms)

                draw_hands(frame, hand_result)
                draw_hud(frame, sign, hint, sample_idx, NUM_SAMPLES,
                         "rec", hand_result, frames_done=frame_idx + 1)
                cv2.imshow("Recoleccion", frame)
                cv2.waitKey(1)

                sequence.append(extract_features(hand_result))

            np.save(save_path, np.array(sequence))
            print(f"  ✓ {sign}/{sample_idx}.npy")
            sample_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    hand_det.close()
    print("\n✅ Recolección completa.")

if __name__ == "__main__":
    collect()