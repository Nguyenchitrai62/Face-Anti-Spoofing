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

def extract_faces_from_frame(frame):
    results = model(frame)
    face_vectors = []
    
    for box in results[0].boxes:
        conf = box.conf[0].item()
        if conf < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue
        face_vector = resize_with_padding(face, TARGET_SIZE)
        face_vectors.append(face_vector)
    
    return face_vectors

def get_face_label():
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ chính
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
    capturing = False  # Biến kiểm soát trạng thái lấy khuôn mặt

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể lấy khung hình từ camera")
            break
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Thoát
            break
        elif key == ord('t'):  # Bắt đầu trích xuất khuôn mặt
            capturing = True
            print("Bắt đầu lấy khuôn mặt...")

        faces = extract_faces_from_frame(frame)
        
        # Hiển thị số ảnh đã lấy trên màn hình
        text = f"press 't' to start count: {count}/10"
        cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Camera", frame)

        if capturing and count < 10:
            if faces:
                face_vectors = np.array(faces, dtype=np.float32)
                faiss.normalize_L2(face_vectors)
                index.add(face_vectors)
                labels.extend([f"{label}"] * len(faces))
                count += 1
                print(f"Ảnh {count}/10 đã lưu với nhãn {label}.")

    cap.release()
    cv2.destroyAllWindows()
    
    # Lưu lại FAISS index và labels
    faiss.write_index(index, index_file)
    np.save(labels_file, labels)
    print("Cập nhật và lưu FAISS index cùng nhãn thành công.")

capture_and_store_faces()