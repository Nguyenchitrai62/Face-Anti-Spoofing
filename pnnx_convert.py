import subprocess

# === Tên model ONNX cần convert ===
onnx_model = "yolov8n-face.onnx"  # Đổi tên file nếu cần

# === Gọi lệnh pnnx ===
cmd = ["./pnnx/pnnx", onnx_model]

# === Thực thi lệnh ===
subprocess.run(cmd)
