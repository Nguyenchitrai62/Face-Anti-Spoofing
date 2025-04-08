import cv2
import torch
import threading
import queue
import numpy as np
import time
from PIL import Image
from scipy.spatial.distance import cosine
from ultralytics import YOLO
import onnxruntime as ort
import faiss

# Load FAISS index và nhãn
index = faiss.read_index("face_index.bin")
labels = np.load("face_labels.npy")

def find_face_label(query_vector, threshold=0.7):
    score_max = -1
    query_vector = query_vector.reshape(-1)
    
    for i in range(index.ntotal):  
        stored_vector = np.zeros((512,), dtype=np.float32)
        index.reconstruct(i, stored_vector)  

        similarity_score = 1 - cosine(query_vector, stored_vector)  
        
        if similarity_score > threshold:
            return labels[i], similarity_score  
        if similarity_score > score_max:
            score_max = similarity_score
    return "unknown", score_max 

# Load model YOLOv8-Face
model_yolo = YOLO("yolov8n-face.pt")

# Load ArcFace model
arcface_model = "w600k_r50.onnx"
session = ort.InferenceSession(arcface_model, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

def resize_with_padding(image, target_size=(112, 112)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    delta_w = target_size[1] - new_w
    delta_h = target_size[0] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)  
    padded = np.transpose(padded, (2, 0, 1))  
    padded = (padded - 127.5) / 127.5 
    padded = np.expand_dims(padded, axis=0).astype(np.float32) 
    
    padded = session.run(None, {input_name: padded})[0]
    return padded.flatten()

# Load mô hình Anti-Spoofing
model_antispoof = torch.jit.load("AENet.pt", map_location=torch.device('cpu'))
model_antispoof.eval()

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    transform = torch.nn.Sequential(
        torch.nn.Upsample(size=(224, 224), mode="bilinear"),
        torch.nn.Identity()
    )
    img = transform(torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0)
    return img

def predict_frame(frame):
    img_tensor = preprocess_frame(frame)
    with torch.no_grad():
        output = model_antispoof(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1).numpy()[0]
        label = "Spoof" if probabilities[1] > probabilities[0] else "Live"
        return label, probabilities[0], probabilities[1]

# **Hàng đợi khuôn mặt**
face_queue = queue.Queue()

def process_faces():
    while True:
        if not face_queue.empty():
            data = face_queue.get()
            frame, x1, y1, x2, y2 = data

            # Cắt khuôn mặt
            face = frame[y1:y2, x1:x2]

            # Nhận diện ArcFace
            face_embedding = resize_with_padding(face)
            person_name, score = find_face_label(face_embedding, 0.7)

            # Kiểm tra live/spoof
            label, prob_live, prob_spoof = predict_frame(face)

            # Hiển thị thông tin
            color = (0, 255, 0) if prob_live > 0.95 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{person_name} ({score:.2f}) - {label}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        time.sleep(0.01)  # Để tránh CPU quá tải

# **Chạy thread xử lý khuôn mặt**
face_thread = threading.Thread(target=process_faces, daemon=True)
face_thread.start()

def run_camera():
    cap = cv2.VideoCapture(2)
    
    if not cap.isOpened():
        raise ValueError("Không thể mở camera. Kiểm tra xem camera có hoạt động không.")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ camera.")
            break

        # Đo thời gian YOLO
        yolo_start = time.time()
        results = model_yolo(frame)  
        yolo_time = time.time() - yolo_start  # Thời gian chạy YOLO

        for box in results[0].boxes:
            conf = box.conf[0].item()
            if conf < 0.7:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            face_queue.put((frame.copy(), x1, y1, x2, y2))  

        # Tính FPS
        fps = 1.0 / (time.time() - start_time)

        # Hiển thị thời gian YOLO
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"YOLO time: {yolo_time:.3f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Face Recognition System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera()
