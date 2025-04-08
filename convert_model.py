import subprocess

# Đường dẫn đến file ONNX
onnx_model = "w600k_r50.onnx"

# Đường dẫn đến công cụ onnx2ncnn (đã được tải về từ ncnn-20241226-windows-vs2022.zip)
onnx2ncnn_path = "./ncnn/x64/bin/onnx2ncnn.exe"

# Đường dẫn output cho model NCNN
param_file = "w600k_r50.param"
bin_file = "w600k_r50.bin"

# Lệnh chuyển đổi
command = [onnx2ncnn_path, onnx_model, param_file, bin_file]

# Thực thi lệnh
try:
    subprocess.run(command, check=True)
    print(f"✅ Chuyển đổi thành công: {param_file}, {bin_file}")
except subprocess.CalledProcessError as e:
    print(f"❌ Lỗi khi chuyển đổi: {e}")
