import cv2
import numpy as np
import faiss
from ultralytics import YOLO
import onnxruntime as ort
import time
import tkinter as tk
from tkinter import simpledialog
import os

CONFIDENCE_THRESHOLD = 0.7
TARGET_SIZE = (112, 112)
SQUARE_SIZE = 300  # Kích thước khung vuông ở giữa
CENTER_THRESHOLD = 0.1  # Ngưỡng để xác định khuôn mặt ở trung tâm
SIZE_THRESHOLD = 0.1  # Ngưỡng để xác định kích thước khuôn mặt phù hợp

# Load model YOLOv8-Face
model = YOLO("yolov8n-face.pt")

# Load ArcFace model
arcface_model = "w600k_r50.onnx"
session = ort.InferenceSession(arcface_model, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# Load FAISS index và labels nếu đã tồn tại
D = 512
index_file = "face_index.bin"
labels_file = "face_labels.npy"
index = None
labels = []

if os.path.exists(index_file) and os.path.exists(labels_file):
    print("Tải FAISS index và labels từ file...")
    index = faiss.read_index(index_file)
    labels = list(np.load(labels_file, allow_pickle=True))
else:
    print("Tạo FAISS index mới...")
    index = faiss.IndexFlatL2(D)

def align_face_by_eyes(face, left_eye, right_eye):
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    aligned = cv2.warpAffine(face, rot_mat, (face.shape[1], face.shape[0]), flags=cv2.INTER_LINEAR)
    return aligned

def resize_with_padding(image, target_size=(112, 112)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    delta_w, delta_h = target_size[1] - new_w, target_size[0] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    padded = np.transpose(padded, (2, 0, 1))
    padded = (padded - 127.5) / 127.5
    padded = np.expand_dims(padded, axis=0).astype(np.float32)
    features = session.run(None, {input_name: padded})[0]
    return features.flatten()

def is_face_approximates_square(box, frame_shape, square_size):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    face_center_x = (x1 + x2) / 2
    face_center_y = (y1 + y2) / 2
    frame_center_x = frame_shape[1] / 2
    frame_center_y = frame_shape[0] / 2
    
    # Kiểm tra vị trí trung tâm
    center_distance_x = abs(face_center_x - frame_center_x) / frame_shape[1]
    center_distance_y = abs(face_center_y - frame_center_y) / frame_shape[0]
    is_centered = center_distance_x < CENTER_THRESHOLD and center_distance_y < CENTER_THRESHOLD
    
    # Kiểm tra kích thước
    face_size = max(x2 - x1, y2 - y1)
    size_ratio = abs(face_size - square_size) / square_size
    is_size_match = size_ratio < SIZE_THRESHOLD
    
    return is_centered and is_size_match

def draw_center_square(frame, square_size):
    h, w = frame.shape[:2]
    top_left = (w//2 - square_size//2, h//2 - square_size//2)
    bottom_right = (w//2 + square_size//2, h//2 + square_size//2)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)  # Màu vàng
    return frame

def extract_faces_from_frame(frame):
    results = model(frame)
    face_vectors = []
    valid_face = False

    for box, kps in zip(results[0].boxes, results[0].keypoints.xy):
        conf = box.conf[0].item()
        if conf < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Vẽ bounding box xanh lá cây cho mọi khuôn mặt
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Kiểm tra xem khuôn mặt có xấp xỉ khung vuông vàng không
        if is_face_approximates_square(box, frame.shape, SQUARE_SIZE):
            valid_face = True
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # Xoay ảnh face nếu có keypoints
            if len(kps) >= 2:
                left_eye = kps[0]
                right_eye = kps[1]
                left_eye = (int(left_eye[0]) - x1, int(left_eye[1]) - y1)
                right_eye = (int(right_eye[0]) - x1, int(right_eye[1]) - y1)
                face = align_face_by_eyes(face, left_eye, right_eye)

            face_vector = resize_with_padding(face, TARGET_SIZE)
            face_vectors.append(face_vector)

    return face_vectors, valid_face

def get_face_label():
    root = tk.Tk()
    root.withdraw()
    label = simpledialog.askstring("Face Label", "Enter face label:")
    return label

def capture_and_store_faces():
    label = get_face_label()
    if not label:
        print("No label entered. Exiting...")
        exit()

    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Không thể mở camera")
        return

    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể lấy khung hình từ camera")
            break

        # Vẽ khung vuông vàng ở giữa
        frame = draw_center_square(frame, SQUARE_SIZE)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        faces, valid_face = extract_faces_from_frame(frame)
        text = f"Count: {count}/10 - Align face with yellow square"
        cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Camera", frame)

        if valid_face and count < 10 and faces:
            face_vectors = np.array(faces, dtype=np.float32)
            faiss.normalize_L2(face_vectors)
            index.add(face_vectors)
            labels.extend([f"{label}"] * len(faces))
            count += 1
            print(f"Ảnh {count}/10 đã lưu với nhãn {label}.")
            # Đợi một chút để tránh lưu nhiều ảnh liên tiếp
            # time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()

    faiss.write_index(index, index_file)
    np.save(labels_file, labels)
    print("Cập nhật và lưu FAISS index cùng nhãn thành công.")

capture_and_store_faces()