import cv2
import time
import sys
import os

# Append current directory to path to ensure src is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.detector import FaceDetector
from src.core.recognizer import FaceRecognizer
from src.core.liveness import LivenessDetector
from src.utils.image_utils import align_face_by_eyes
import src.config.settings as settings

def run_camera():
    # Initialize Models
    print("Loading models...")
    try:
        detector = FaceDetector(settings.YOLO_MODEL_PATH)
        recognizer = FaceRecognizer(settings.ARCFACE_MODEL_PATH, 
                                    settings.FACE_INDEX_PATH, 
                                    settings.FACE_LABELS_PATH)
        liveness_detector = LivenessDetector(settings.AENET_MODEL_PATH)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    cap = cv2.VideoCapture(2) # Defaulting to camera index 2 as per original code
    if not cap.isOpened():
        print("Camera 2 not found, trying 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open any camera.")
            return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame from camera.")
            break

        start_time = time.time()
        
        # Detection
        results, yolo_time = detector.detect(frame, settings.DEFAULT_CONFIDENCE)

        face_process_time = 0
        find_label_time = 0
        predict_time = 0

        for idx, result in enumerate(results):
            # Already filtered by conf inside detect but check again if needed
            if result['conf'] < settings.DEFAULT_CONFIDENCE:
                continue

            x1, y1, x2, y2 = result['xyxy']
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            face = frame[y1:y2, x1:x2].copy()
            if face.size == 0: continue

            # Alignment
            aligned_face = None
            if len(result['keypoints']) >= 2:
                 # Keypoints: [left_eye, right_eye, nose, mouth_left, mouth_right]
                left_eye, right_eye = result['keypoints'][0], result['keypoints'][1]
                
                # Relativize to face crop
                left_eye_rel = (left_eye[0] - x1, left_eye[1] - y1)
                right_eye_rel = (right_eye[0] - x1, right_eye[1] - y1)
                
                # Check bounds for eyes to avoid errors
                aligned_face = align_face_by_eyes(face, left_eye_rel, right_eye_rel)

            face_for_embedding = aligned_face if aligned_face is not None else face

            # Recognition (Embedding + ID)
            face_start = time.time()
            face_embedding = recognizer.get_embedding(face_for_embedding)
            face_process_time += time.time() - face_start

            find_label_start = time.time()
            person_name, score = recognizer.identify_face(face_embedding, settings.DEFAULT_CONFIDENCE)
            find_label_time += time.time() - find_label_start

            # Liveness
            predict_start = time.time()
            label, prob_live, prob_spoof = liveness_detector.predict(face_for_embedding)
            predict_time += time.time() - predict_start

            # Drawing
            color = (0, 255, 0) if prob_live > 0.95 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            text = f"Live: {prob_live:.2f} | {person_name}:({score:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Landmarks
            for idx_kp, (lx, ly) in enumerate(result['keypoints']):
                color_kp = (0, 255, 255)
                if idx_kp in [0, 1]: color_kp = (255, 0, 0)
                elif idx_kp == 2: color_kp = (0, 255, 0)
                elif idx_kp in [3, 4]: color_kp = (0, 0, 255)
                cv2.circle(frame, (lx, ly), 3, color_kp, -1)
            
             # Save logic
            key = cv2.waitKey(1) & 0xFF
            if key == ord('t'):
                 cv2.imwrite(f"face_crop_{idx}.jpg", face)
                 if aligned_face is not None:
                      cv2.imwrite(f"face_aligned_{idx}.jpg", aligned_face)
                 print(f"Saved crop {idx}")

        total_time = time.time() - start_time
        fps = 1.0 / total_time if total_time > 0 else 0

        # Info display
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"YOLO: {yolo_time:.3f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"ArcFace: {face_process_time:.3f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"FAISS: {find_label_time:.3f}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Liveness: {predict_time:.3f}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Face Anti-Spoofing", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera()
