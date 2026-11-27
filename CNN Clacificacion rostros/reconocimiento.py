import os
import sys
import json
import time
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# ==============================================================================
# CONFIGURACIÓN Y CONSTANTES
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "modelo_facial_finetuned.keras"
CLASSES_FILENAME = "clases.json"
TASK_FILENAME = "face_landmarker.task"

MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)
CLASSES_PATH = os.path.join(BASE_DIR, CLASSES_FILENAME)
TASK_PATH = os.path.join(BASE_DIR, TASK_FILENAME)

IMG_SIZE = (224, 224)
PADDING_PERCENT = 0.15      
UMBRAL_CONFIANZA = 0.65
CAMERA_INDEX = 0

# Se mantiene la red neuronal ejecutándose cada 5 frames
PROCESS_EVERY_N_FRAMES = 5

# Índices clave de MediaPipe Face Mesh para alineación
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ==============================================================================
# INICIALIZACIÓN Y VALIDACIÓN DE ARCHIVOS
# ==============================================================================
def verificar_archivos():
    faltantes = [p for p in [MODEL_PATH, CLASSES_PATH, TASK_PATH] if not os.path.exists(p)]
    if faltantes:
        print("ERROR: Faltan los siguientes archivos:")
        for f in faltantes: print(f"  - {f}")
        sys.exit(1)

verificar_archivos()

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    data_clases = json.load(f)
    if isinstance(data_clases, dict):
        if all(isinstance(v, int) for v in data_clases.values()):
            clases = [k for k, v in sorted(data_clases.items(), key=lambda x: x[1])]
        else:
            clases = [data_clases[str(i)] for i in range(len(data_clases))]
    else:
        clases = data_clases

print("Cargando modelo MobileNetV2...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Calentando el modelo (Warm-up)...")
dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
_ = model(dummy_input, training=False)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=5
)

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================
def align_and_crop(image, landmarks_list, pad_pct):
    h, w, _ = image.shape
    
    left_eye_pts = np.array([landmarks_list[i] for i in LEFT_EYE])
    right_eye_pts = np.array([landmarks_list[i] for i in RIGHT_EYE])
    
    left_center = left_eye_pts.mean(axis=0)
    right_center = right_eye_pts.mean(axis=0)
    
    dY = right_center[1] - left_center[1]
    dX = right_center[0] - left_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    face_center_raw = np.array(landmarks_list).mean(axis=0)
    face_center = (float(face_center_raw[0]), float(face_center_raw[1]))
    
    M = cv2.getRotationMatrix2D(face_center, angle, 1.0)
    
    rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    ones = np.ones((len(landmarks_list), 1))
    pts3d = np.hstack([landmarks_list, ones])
    transformed_pts = M.dot(pts3d.T).T
    
    min_x = int(np.min(transformed_pts[:, 0]))
    max_x = int(np.max(transformed_pts[:, 0]))
    min_y = int(np.min(transformed_pts[:, 1]))
    max_y = int(np.max(transformed_pts[:, 1]))
    
    face_w = max_x - min_x
    face_h = max_y - min_y
    
    base_side = max(face_w, face_h)
    cx = int((min_x + max_x) / 2)
    cy = int((min_y + max_y) / 2) - int(face_h * 0.05)
    
    half_side = int((base_side / 2.0) * (1.0 + pad_pct))
    
    start_x = cx - half_side
    end_x = cx + half_side
    start_y = cy - half_side
    end_y = cy + half_side
    
    if start_x < 0 or start_y < 0 or end_x > w or end_y > h:
        return None  
        
    return rotated_img[start_y:end_y, start_x:end_x]

# ==============================================================================
# BUCLE PRINCIPAL
# ==============================================================================
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir la cámara {CAMERA_INDEX}")
        sys.exit(1)
        
    # MEJORA CLAVE: Bajar la resolución a 640x480 (reduce el trabajo del procesador)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nIniciando cámara... Presiona 'q' o 'ESC' para salir.")
    
    last_timestamp_ms = 0
    frame_counter = 0
    last_predictions = {} 
    
    try:
        landmarker = FaceLandmarker.create_from_options(options)
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            frame_counter += 1
            
            run_cnn = (frame_counter % PROCESS_EVERY_N_FRAMES == 0)
            
            frame_h, frame_w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            timestamp_ms = int(time.monotonic() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                if run_cnn:
                    #print("\033[H\033[J", end="") 
                    #print(f"--- DIAGNÓSTICO EN VIVO ({len(result.face_landmarks)} rostro/s) ---")
                    last_predictions.clear()

                for idx, face_landmarks in enumerate(result.face_landmarks):
                    landmarks_px = [[int(lm.x * frame_w), int(lm.y * frame_h)] for lm in face_landmarks]
                    
                    min_x = min(p[0] for p in landmarks_px)
                    max_x = max(p[0] for p in landmarks_px)
                    min_y = min(p[1] for p in landmarks_px)
                    max_y = max(p[1] for p in landmarks_px)

                    if run_cnn:
                        cropped_bgr = align_and_crop(frame, landmarks_px, PADDING_PERCENT)
                        
                        if cropped_bgr is not None:
                            cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
                            roi_resized = cv2.resize(cropped_rgb, IMG_SIZE)
                            roi_batch = np.expand_dims(np.array(roi_resized, dtype=np.float32), axis=0)

                            predicciones = model(roi_batch, training=False).numpy()[0]
                            indice_clase = np.argmax(predicciones)
                            confianza = predicciones[indice_clase]

                            #print(f"\nRostro {idx + 1}:")
                            #for i, conf in enumerate(predicciones):
                            #    marcador = ">>> " if i == indice_clase else "    "
                            #    print(f"{marcador}{clases[i]}: {conf*100:>5.2f}%")
                            
                            if confianza >= UMBRAL_CONFIANZA:
                                etiqueta = f"{clases[indice_clase]} ({confianza*100:.1f}%)"
                                color_caja = (0, 255, 0)
                            else:
                                etiqueta = f"Desconocido ({confianza*100:.1f}%)"
                                color_caja = (0, 0, 255)
                            
                            last_predictions[idx] = (etiqueta, color_caja)
                        else:
                            last_predictions[idx] = ("Descartado (bordes)", (0, 165, 255))
                            #print(f"\nRostro {idx + 1}: Recorte descartado (Fuera de límites)")
                    
                    if idx in last_predictions:
                        etiqueta, color_caja = last_predictions[idx]
                        label_y = max(min_y, 25)
                        
                        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color_caja, 2)
                        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        
                        cv2.rectangle(frame, (min_x, label_y - 25), (min_x + tw, label_y), color_caja, -1)
                        cv2.putText(frame, etiqueta, (min_x, label_y - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow('Reconocimiento Facial (Alineado)', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
                
    finally:
        print("\nLiberando recursos...")
        if 'landmarker' in locals():
            landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()