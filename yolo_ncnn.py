import cv2
import time
import ncnn
import numpy as np

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


if __name__ == "__main__":
    image_path = "dataset/ori/Akshay Kumar/anh_10.jpg"
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Không thể đọc ảnh")

    result, t = detect_faces_ncnn(img)

    # Hiển thị thời gian trên ảnh
    cv2.putText(result, f"{t * 1000:.1f} ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Face - NCNN", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
