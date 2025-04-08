import cv2
import numpy as np
import ncnn
from scipy.spatial.distance import cosine
import time

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
    img = (img - 127.5) / 127.5  # normalize
    return img.astype(np.float32)

def extract_ncnn_features(net, image_path):
    img = preprocess_image(image_path)
    img = np.transpose(img, (2, 0, 1)).flatten()  # CHW → 1D
    in_blob = ncnn.Mat(112, 112, 3)

    for i in range(len(img)):
        in_blob[i] = img[i]

    extractor = net.create_extractor()

    start = time.perf_counter()
    extractor.input("input.1", in_blob)
    ret, out_blob = extractor.extract("683")  # ArcFace output node
    end = time.perf_counter()

    print(f"[NCNN] Thời gian chạy: {(end - start)*1000:.2f} ms")
    return np.array(out_blob).flatten()

def compare_faces_ncnn(image1, image2, net):
    feat1 = extract_ncnn_features(net, image1)
    feat2 = extract_ncnn_features(net, image2)
    similarity = 1 - cosine(feat1, feat2)
    print("\n[INFO] Feature 1 (NCNN):", feat1[:5])
    print(f"\nMức độ tương đồng (NCNN): {similarity:.4f}")
    return similarity

# Load model NCNN
ncnn_param_path = "w600k_r50.param"
ncnn_bin_path = "w600k_r50.bin"
net = load_ncnn_model(ncnn_param_path, ncnn_bin_path)

# So sánh ảnh
image_path1 = "dataset/train/Akshay Kumar/anh_11.jpg"
image_path2 = "dataset/train/Akshay Kumar/anh_27.jpg"
similarity = compare_faces_ncnn(image_path1, image_path2, net)
