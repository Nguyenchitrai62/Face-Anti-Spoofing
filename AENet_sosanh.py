import torch
import onnxruntime as ort
import ncnn
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import os
import torch.nn.functional as F
import time

# Load TorchScript model (.pt)
torch_model = torch.jit.load("AENet.pt", map_location=torch.device('cpu'))
torch_model.eval()

# Load ONNX model
onnx_session = ort.InferenceSession("AENet.onnx", providers=["CPUExecutionProvider"])
onnx_input_name = onnx_session.get_inputs()[0].name

# Load NCNN model
def load_ncnn_model():
    ncnn_net = ncnn.Net()
    ncnn_net.load_param("AENet.ncnn.param")
    ncnn_net.load_model("AENet.ncnn.bin")
    return ncnn_net

# Tiền xử lý ảnh
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)  # shape (1, 3, 224, 224)
    return img_tensor

# Dự đoán với NCNN
def test_inference_ncnn(image_path, ncnn_net):
    image = preprocess_image(image_path)
    ex = ncnn_net.create_extractor()
    ex.input("in0", ncnn.Mat(image.squeeze(0).numpy()).clone())
    _, out0 = ex.extract("out0")
    out = torch.from_numpy(np.array(out0)).unsqueeze(0)
    probs = F.softmax(out, dim=1)
    return probs

# So sánh kết quả và thời gian giữa 3 model
def compare_models(image_path, stats):
    input_tensor = preprocess_image(image_path)

    # PyTorch
    start = time.time()
    with torch.no_grad():
        pt_output = torch_model(input_tensor)
        pt_probs = torch.nn.functional.softmax(pt_output, dim=1).numpy()[0]
    pt_time = time.time() - start

    # ONNX
    start = time.time()
    onnx_input = input_tensor.numpy().astype(np.float32)
    onnx_output = onnx_session.run(None, {onnx_input_name: onnx_input})[0]
    onnx_probs = torch.nn.functional.softmax(torch.tensor(onnx_output), dim=1).numpy()[0]
    onnx_time = time.time() - start

    # NCNN
    ncnn_net = load_ncnn_model()
    start = time.time()
    ncnn_probs = test_inference_ncnn(image_path, ncnn_net)[0].numpy()
    ncnn_time = time.time() - start

    # In kết quả
    print(f"\n📸 Ảnh: {os.path.basename(image_path)}")
    print(f"PyTorch - Live: {pt_probs[0]:.4f}, Spoof: {pt_probs[1]:.4f} | ⏱ {pt_time:.4f}s")
    print(f"ONNX    - Live: {onnx_probs[0]:.4f}, Spoof: {onnx_probs[1]:.4f} | ⏱ {onnx_time:.4f}s")
    print(f"NCNN    - Live: {ncnn_probs[0]:.4f}, Spoof: {ncnn_probs[1]:.4f} | ⏱ {ncnn_time:.4f}s")

    # Tính sai khác
    pt_diff = np.abs(pt_probs - ncnn_probs)
    onnx_diff = np.abs(onnx_probs - ncnn_probs)
    print(f"Δ Sai khác giữa PyTorch và NCNN: {pt_diff}, Tổng: {np.sum(pt_diff):.6f}")
    print(f"Δ Sai khác giữa ONNX và NCNN: {onnx_diff}, Tổng: {np.sum(onnx_diff):.6f}")

    # Cập nhật thống kê thời gian
    stats['pytorch'].append(pt_time)
    stats['onnx'].append(onnx_time)
    stats['ncnn'].append(ncnn_time)

# Lặp qua các ảnh và tính trung bình thời gian
def compare_images_in_folder():
    stats = {'pytorch': [], 'onnx': [], 'ncnn': []}
    for i in range(6, 21):
        img_path = f"./dataset/train/Akshay Kumar/anh_{i}.jpg"
        if os.path.exists(img_path):
            compare_models(img_path, stats)
        else:
            print(f"⚠️ Không tìm thấy ảnh: {img_path}")

    # In thời gian trung bình
    print("\n📊 Thời gian suy luận trung bình:")
    print(f"PyTorch: {np.mean(stats['pytorch']):.4f}s")
    print(f"ONNX   : {np.mean(stats['onnx']):.4f}s")
    print(f"NCNN   : {np.mean(stats['ncnn']):.4f}s")

if __name__ == "__main__":
    compare_images_in_folder()
