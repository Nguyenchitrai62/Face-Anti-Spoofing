import cv2
import numpy as np
import time
import faiss
import onnxruntime as ort
from scipy.spatial.distance import cosine
import torchvision.transforms as transforms
from PIL import Image
import torch
from ultralytics import YOLO

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

def detect_faces_pt(img, model, conf_threshold=0.7):
    start_time = time.time()
    results = model.predict(source=img, conf=conf_threshold, verbose=False)
    elapsed = time.time() - start_time
    detections = results[0].boxes
    keypoints_list = results[0].keypoints.xy if results[0].keypoints is not None else None

    results_list = []
    for i, box in enumerate(detections):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])

        # Lấy keypoints nếu có
        if keypoints_list is not None and i < len(keypoints_list):
            kps = keypoints_list[i].cpu().numpy().astype(int)
            keypoints = [(int(x), int(y)) for x, y in kps]
        else:
            keypoints = []

        results_list.append({
            'xyxy': [x1, y1, x2, y2],
            'conf': conf,
            'keypoints': keypoints
        })
    return results_list, elapsed


def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0)
    return img

def predict_frame(session, input_name, frame):
    img_tensor = preprocess_frame(frame)
    onnx_input = img_tensor.numpy().astype(np.float32)
    onnx_output = session.run(None, {input_name: onnx_input})[0]
    probabilities = torch.nn.functional.softmax(torch.tensor(onnx_output), dim=1).numpy()[0]
    label = "Spoof" if probabilities[1] > probabilities[0] else "Live"
    return label, probabilities[0], probabilities[1]

def align_face_by_eyes(image, left_eye, right_eye):
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
    return aligned


def run_camera(model_path="AENet.onnx"):
    session_aenet = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name_aenet = session_aenet.get_inputs()[0].name
    yolo_model = YOLO("yolov8n-face.pt")
    print("Models loaded successfully.")

    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        raise ValueError("Không thể mở camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ camera.")
            break

        start_time = time.time()
        results, yolo_time = detect_faces_pt(frame, yolo_model)

        face_process_time = 0
        find_label_time = 0
        predict_time = 0

        for idx, result in enumerate(results):
            if result['conf'] < 0.7:
                continue

            x1, y1, x2, y2 = map(int, result['xyxy'])
            face = frame[y1:y2, x1:x2].copy()  # copy để lưu ảnh gốc crop

            aligned_face = None
            # Nếu có keypoints thì align theo 2 mắt
            if len(result['keypoints']) >= 2:
                # Keypoints trong YOLOv8-face: [left_eye, right_eye, nose, mouth_left, mouth_right]
                left_eye, right_eye = result['keypoints'][0], result['keypoints'][1]

                # Chuyển về tọa độ relative trong face crop
                left_eye = (left_eye[0] - x1, left_eye[1] - y1)
                right_eye = (right_eye[0] - x1, right_eye[1] - y1)

                aligned_face = align_face_by_eyes(face, left_eye, right_eye)

            # Chọn ảnh để tính embedding: nếu có align thì dùng align, không thì dùng crop gốc
            face_for_embedding = aligned_face if aligned_face is not None else face

            # Embedding
            face_start = time.time()
            face_embedding = resize_with_padding(face_for_embedding)
            face_process_time += time.time() - face_start

            # Nhận dạng
            find_label_start = time.time()
            person_name, score = find_face_label(face_embedding, 0.7)
            find_label_time += time.time() - find_label_start

            # Dự đoán spoofing
            predict_start = time.time()
            label, prob_live, prob_spoof = predict_frame(session_aenet, input_name_aenet, face_for_embedding)
            predict_time += time.time() - predict_start

            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if prob_live > 0.95 else (0, 0, 255), 2)
            text = f"Live: {prob_live:.2f} | {person_name}:({score:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Vẽ landmarks: mắt, mũi, miệng
            for idx_kp, (lx, ly) in enumerate(result.get('keypoints', [])):
                color = (0, 255, 255)
                if idx_kp == 0 or idx_kp == 1:
                    color = (255, 0, 0)  # mắt
                elif idx_kp == 2:
                    color = (0, 255, 0)  # mũi
                elif idx_kp == 3 or idx_kp == 4:
                    color = (0, 0, 255)  # miệng
                cv2.circle(frame, (lx, ly), 3, color, -1)

            # Lưu ảnh khi nhấn 't'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('t'):
                cv2.imwrite(f"face_crop_{idx}.jpg", face)
                if aligned_face is not None:
                    cv2.imwrite(f"face_aligned_{idx}.jpg", aligned_face)
                else:
                    print("Không có ảnh đã align để lưu.")
                print(f"Đã lưu ảnh face_crop_{idx}.jpg và face_aligned_{idx}.jpg nếu có.")


        total_time = time.time() - start_time
        fps = 1.0 / total_time

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"YOLO: {yolo_time:.3f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"arcface: {face_process_time:.3f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"faiss: {find_label_time:.3f}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"live/spoof: {predict_time:.3f}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Face Anti-Spoofing with Landmarks", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera()
