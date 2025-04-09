import cv2
import numpy as np
import time
import onnxruntime as ort
import ncnn
from ultralytics import YOLO

# --- NCNN Inference ---
def detect_faces_ncnn(img):
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.load_param("yolov8n_face.ncnn.param")
    net.load_model("yolov8n_face.ncnn.bin")

    input_size = 640
    h0, w0 = img.shape[:2]

    img_resized = cv2.resize(img, (input_size, input_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_normalized, (2, 0, 1))

    mat = ncnn.Mat(input_size, input_size, 3)
    flat = img_chw.flatten()
    for i in range(flat.shape[0]):
        mat[i] = flat[i]

    ex = net.create_extractor()
    ex.input("in0", mat)

    start_time = time.time()
    _, out = ex.extract("out0")
    elapsed = time.time() - start_time

    out_np = np.frombuffer(out, dtype=np.float32).reshape((out.h, out.w))

    boxes = []
    confidences = []
    for i in range(out_np.shape[1]):
        x, y, w_box, h_box, conf = out_np[0:5, i]
        if conf < 0.5:
            continue

        x1 = int((x - w_box / 2) * w0 / input_size)
        y1 = int((y - h_box / 2) * h0 / input_size)
        x2 = int((x + w_box / 2) * w0 / input_size)
        y2 = int((y + h_box / 2) * h0 / input_size)

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(float(conf))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    result = img.copy()
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        x, y, w_box, h_box = boxes[i]
        cv2.rectangle(result, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
        cv2.putText(result, f"{confidences[i]:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return result, elapsed


# --- ONNX Inference ---
def detect_faces_onnx(img):
    session = ort.InferenceSession("yolov8n-face.onnx")
    input_name = session.get_inputs()[0].name

    input_size = (640, 640)
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
        if conf < 0.5:
            continue
        x, y, w, h = pred[0:4]
        x1 = int((x - w / 2) * img.shape[1] / input_size[0])
        y1 = int((y - h / 2) * img.shape[0] / input_size[1])
        x2 = int((x + w / 2) * img.shape[1] / input_size[0])
        y2 = int((y + h / 2) * img.shape[0] / input_size[1])
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(float(conf))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    result = img.copy()
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        x, y, w, h = boxes[i]
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result, f"{confidences[i]:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return result, elapsed


# --- PyTorch Inference ---
def detect_faces_pt(img):
    model = YOLO("yolov8n-face.pt")
    start_time = time.time()
    results = model(img)[0]
    elapsed = time.time() - start_time

    result = img.copy()
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        if conf < 0.5:
            continue
        cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(result, f"{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return result, elapsed


# --- Run and Compare ---
if __name__ == "__main__":
    image_path = "dataset/ori/Akshay Kumar/anh_10.jpg"
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    img_ncnn, time_ncnn = detect_faces_ncnn(img)
    print(f"[NCNN] Inference time: {time_ncnn:.3f} seconds")

    img_onnx, time_onnx = detect_faces_onnx(img)
    print(f"[ONNX] Inference time: {time_onnx:.3f} seconds")

    img_pt, time_pt = detect_faces_pt(img)
    print(f"[PyTorch] Inference time: {time_pt:.3f} seconds")

    # Ghép ảnh theo chiều dọc
    top_row = np.hstack([img_pt, img_onnx])
    bottom_row = np.hstack([img_ncnn, np.zeros_like(img_ncnn)])
    concat = np.vstack([top_row, bottom_row])

    # Thêm text thời gian
    h, w = img.shape[:2]
    cv2.putText(concat, f"PyTorch: {time_pt:.3f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(concat, f"ONNX: {time_onnx:.3f}s", (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(concat, f"NCNN: {time_ncnn:.3f}s", (10, h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("YOLOv8 Face - PyTorch vs ONNX vs NCNN", concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
