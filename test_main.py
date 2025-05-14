import cv2
import numpy as np
import time
import faiss
import onnxruntime as ort
from scipy.spatial.distance import cosine
import torch
import torchvision.transforms as transforms
from PIL import Image

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

# Load ArcFace model
arcface_model = "w600k_r50.onnx"
session_arcface = ort.InferenceSession(arcface_model, providers=['CPUExecutionProvider'])
input_name_arcface = session_arcface.get_inputs()[0].name

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
    
    padded = session_arcface.run(None, {input_name_arcface: padded})[0]
    return padded.flatten()

# Hàm phát hiện khuôn mặt sử dụng YOLO ONNX
def detect_faces_onnx(img, session, input_name, input_size=(640, 640), conf_threshold=0.7):
    img_resized = cv2.resize(img, input_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    input_tensor = np.expand_dims(img_transposed, axis=0)

    start_time = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    elapsed = time.time() - start_time

    preds = outputs[0]
    preds = np.transpose(preds, (0, 2, 1))[0]  # [8400, 20]
    boxes = []
    confidences = []

    for pred in preds:
        conf = pred[4]
        if conf < conf_threshold:
            continue
        x, y, w, h = pred[0:4]
        x1 = int((x - w / 2) * img.shape[1] / input_size[0])
        y1 = int((y - h / 2) * img.shape[0] / input_size[1])
        x2 = int((x + w / 2) * img.shape[1] / input_size[0])
        y2 = int((y + h / 2) * img.shape[0] / input_size[1])
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(float(conf))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)
    results = []
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        x, y, w, h = boxes[i]
        results.append({
            'xyxy': [x, y, x + w, y + h],
            'conf': confidences[i]
        })
    return results, elapsed

# Tiền xử lý frame từ camera
def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0) 
    return img

# Dự đoán và trả về nhãn
def predict_frame(model, frame):
    img_tensor = preprocess_frame(frame)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1).numpy()[0]
        label = "Spoof" if probabilities[1] > probabilities[0] else "Live"
        return label, probabilities[0], probabilities[1]

def run_camera(model_path="AENet.pt"):
    # Load mô hình đã lưu
    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    print("Model loaded successfully.")

    # Load YOLO ONNX model
    session_yolo = ort.InferenceSession("yolov8n-face.onnx", providers=['CPUExecutionProvider'])
    input_name_yolo = session_yolo.get_inputs()[0].name

    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        raise ValueError("Không thể mở camera. Kiểm tra xem camera có hoạt động không.")

    while True:
        tick_start = cv2.getTickCount()

        start_time = time.time()  # Bắt đầu đo toàn bộ vòng lặp
        
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ camera.")
            break

        # Đo thời gian YOLO ONNX detect
        yolo_start = time.time()
        results, yolo_time = detect_faces_onnx(frame, session_yolo, input_name_yolo)
        yolo_time = time.time() - yolo_start

        face_process_time = 0
        find_label_time = 0
        predict_time = 0

        for result in results:
            conf = result['conf']
            if conf < 0.7:
                continue

            x1, y1, x2, y2 = map(int, result['xyxy'])
            face = frame[y1:y2, x1:x2]  

            # Đo thời gian resize + ArcFace
            face_start = time.time()
            face_embedding = resize_with_padding(face)
            face_process_time += time.time() - face_start

            # Đo thời gian tìm nhãn với FAISS
            find_label_start = time.time()
            person_name, score = find_face_label(face_embedding, 0.7)
            find_label_time += time.time() - find_label_start

            # Đo thời gian chạy mô hình Anti-Spoofing
            predict_start = time.time()
            label, prob_live, prob_spoof = predict_frame(model, face)
            predict_time += time.time() - predict_start

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if prob_live > 0.95 else (0, 0, 255), 2)
            text = f"Live: {prob_live:.2f} | {person_name}:({score:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Đo tổng thời gian
        total_time = time.time() - start_time
        fps = 1.0 / total_time

        # Hiển thị thời gian đo được
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"YOLO: {yolo_time:.3f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"resize & arcface: {face_process_time:.3f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"faiss: {find_label_time:.3f}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"live/spoof: {predict_time:.3f}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Face Anti-Spoofing", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera()