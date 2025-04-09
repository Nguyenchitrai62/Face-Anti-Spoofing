import cv2
import numpy as np
import onnxruntime as ort
from scipy.spatial.distance import cosine
import ncnn
import time

def load_onnx_model(model_path):
    return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

def load_ncnn_model(param_path, bin_path):
    net = ncnn.Net()
    net.load_param(param_path)
    net.load_model(bin_path)
    return net

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể load ảnh: {image_path}")
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img - 127.5) / 127.5
    return img.astype(np.float32)

def extract_onnx_features(session, image_path):
    img = preprocess_image(image_path)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    input_name = session.get_inputs()[0].name

    start = time.perf_counter()
    features = session.run(None, {input_name: img})[0]
    end = time.perf_counter()

    print(f"[ONNX] Thời gian chạy: {(end - start)*1000:.2f} ms")
    return features.flatten()

def extract_ncnn_features(net, image_path):
    img = preprocess_image(image_path)
    img = np.transpose(img, (2, 0, 1))
    img = img.flatten().astype(np.float32)

    in_blob = ncnn.Mat(112, 112, 3)
    for i in range(len(img)):
        in_blob[i] = img[i]

    extractor = net.create_extractor()

    start = time.perf_counter()
    extractor.input("input.1", in_blob)
    ret, out_blob = extractor.extract("683")
    end = time.perf_counter()

    print(f"[NCNN] Thời gian chạy: {(end - start)*1000:.2f} ms")
    return np.array(out_blob).flatten()

def compare_faces(image1, image2, session, net):
    feat1_onnx = extract_onnx_features(session, image1)
    feat2_onnx = extract_onnx_features(session, image2)
    similarity_onnx = 1 - cosine(feat1_onnx, feat2_onnx)

    feat1_ncnn = extract_ncnn_features(net, image1)
    feat2_ncnn = extract_ncnn_features(net, image2)
    similarity_ncnn = 1 - cosine(feat1_ncnn, feat2_ncnn)

    print("\n[INFO] ONNX Feature 1:", feat1_onnx[:5])
    print("[INFO] NCNN Feature 1:", feat1_ncnn[:5])

    return similarity_onnx, similarity_ncnn

# Load models
onnx_model_path = "w600k_r50.onnx"
ncnn_param_path = "w600k_r50.param"
ncnn_bin_path = "w600k_r50.bin"
session = load_onnx_model(onnx_model_path)
net = load_ncnn_model(ncnn_param_path, ncnn_bin_path)

# Test
image_path1 = "dataset/train/Akshay Kumar/anh_11.jpg"
image_path2 = "dataset/train/Akshay Kumar/anh_27.jpg"
similarity_onnx, similarity_ncnn = compare_faces(image_path1, image_path2, session, net)

print(f"\nMức độ tương đồng (ONNX): {similarity_onnx:.4f}")
print(f"Mức độ tương đồng (NCNN): {similarity_ncnn:.4f}")
