import cv2
import numpy as np
import faiss
import time
import tkinter as tk
from tkinter import simpledialog
import os
import sys

# Path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.detector import FaceDetector
from src.core.recognizer import FaceRecognizer
from src.utils.image_utils import align_face_by_eyes
import src.config.settings as settings

CONFIDENCE_THRESHOLD = 0.7
SQUARE_SIZE = 300 
CENTER_THRESHOLD = 0.1 
SIZE_THRESHOLD = 0.1 
EXTRACT_INTERVAL = 0.5 

def is_face_approximates_square(x1, y1, x2, y2, frame_shape, square_size):
    face_center_x = (x1 + x2) / 2
    face_center_y = (y1 + y2) / 2
    frame_center_x = frame_shape[1] / 2
    frame_center_y = frame_shape[0] / 2
    
    # Check center
    center_distance_x = abs(face_center_x - frame_center_x) / frame_shape[1]
    center_distance_y = abs(face_center_y - frame_center_y) / frame_shape[0]
    is_centered = center_distance_x < CENTER_THRESHOLD and center_distance_y < CENTER_THRESHOLD
    
    # Check size
    face_size = max(x2 - x1, y2 - y1)
    size_ratio = abs(face_size - square_size) / square_size
    is_size_match = size_ratio < SIZE_THRESHOLD
    
    return is_centered and is_size_match

def draw_center_square(frame, square_size):
    h, w = frame.shape[:2]
    top_left = (w//2 - square_size//2, h//2 - square_size//2)
    bottom_right = (w//2 + square_size//2, h//2 + square_size//2)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)  # Yellow
    return frame

def get_face_label():
    root = tk.Tk()
    root.withdraw()
    label = simpledialog.askstring("Face Label", "Enter face label:")
    return label

def capture_and_store_faces():
    print(f"Loading models from {settings.MODELS_DIR}...")
    detector = FaceDetector(settings.YOLO_MODEL_PATH)
    recognizer = FaceRecognizer(settings.ARCFACE_MODEL_PATH, 
                                settings.FACE_INDEX_PATH, 
                                settings.FACE_LABELS_PATH)
    
    # Handle Index and Labels initialization manually for registration
    index = recognizer.index
    labels = []
    if index is None:
        print("Initializing new index")
        index = faiss.IndexFlatL2(512)
    else:
        # Check if index is loaded correctly by printing ntotal
        print(f"Loaded index with {index.ntotal} vectors")

    if recognizer.labels is not None:
        labels = recognizer.labels.tolist()
    
    label = get_face_label()
    if not label:
        print("No label entered. Exiting...")
        return

    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
         cap = cv2.VideoCapture(0)
         if not cap.isOpened():
            print("Cannot open camera")
            return

    count = 0
    last_extract_time = 0 
    print("Starting capture loop...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = draw_center_square(frame, SQUARE_SIZE)
        
        # Detect
        # Lower confidence slightly for detection if needed, but 0.7 is fine
        results, _ = detector.detect(frame, CONFIDENCE_THRESHOLD)
        
        valid_face_found = False
        face_vectors = []

        for result in results:
            x1, y1, x2, y2 = result['xyxy']
            # Draw green box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if is_face_approximates_square(x1, y1, x2, y2, frame.shape, SQUARE_SIZE):
                 valid_face_found = True
                 face = frame[y1:y2, x1:x2].copy()
                 if face.size == 0: continue
                 
                 # Align
                 if len(result['keypoints']) >= 2:
                    left_eye, right_eye = result['keypoints'][0], result['keypoints'][1]
                    left_eye_rel = (left_eye[0] - x1, left_eye[1] - y1)
                    right_eye_rel = (right_eye[0] - x1, right_eye[1] - y1)
                    face = align_face_by_eyes(face, left_eye_rel, right_eye_rel)
                
                 # Get embedding
                 embedding = recognizer.get_embedding(face)
                 face_vectors.append(embedding)

        text = f"Count: {count}/10 - Align face with yellow square"
        cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Camera", frame)
        
        current_time = time.time()
        if valid_face_found and count < 10 and face_vectors and (current_time - last_extract_time >= EXTRACT_INTERVAL):
            # Normalize and add
            vecs = np.array(face_vectors, dtype=np.float32)
            # Normalize logic from original code?
            # Original: faiss.normalize_L2(face_vectors)
            faiss.normalize_L2(vecs)
            index.add(vecs)
            labels.extend([label] * len(vecs))
            count += 1
            last_extract_time = current_time
            print(f"Captured {count}/10 for {label}")
        
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 10:
            break

    cap.release()
    cv2.destroyAllWindows()

    if count > 0:
        faiss.write_index(index, settings.FACE_INDEX_PATH)
        np.save(settings.FACE_LABELS_PATH, labels)
        print("Updated FAISS index and labels.")
    else:
        print("No faces captured.")

if __name__ == "__main__":
    capture_and_store_faces()
