# Face Anti-Spoofing & Recognition Project

## Cài đặt (Installation)

1. **Cài đặt PyTorch:**
   Cài đặt `torch` và `torchvision` (phiên bản hỗ trợ CUDA 12.1):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Cài đặt các thư viện còn lại:**
   ```bash
   pip install -r requirements.txt
   ```

## Cấu trúc dự án (Project Structure)
Dự án đã được tái cấu trúc để gọn gàng và dễ mở rộng hơn:

- `src/`: Chứa mã nguồn lõi (Core logic, Utils, Config).
- `scripts/`: Chứa các scripts chạy tính năng (Đăng ký, Test, Convert).
- `models/`: Thư mục chứa các file weights (onnx, pt).
- `main.py`: Chương trình chính.

## Hướng dẫn sử dụng (Usage)

### 1. Chạy nhận diện (Main Program)
Khởi động camera để nhận diện khuôn mặt và kiểm tra liveness (người thật/giả):
```bash
python main.py
```
- Phím `t`: Lưu ảnh crop khuôn mặt hiện tại.
- Phím `q`: Thoát chương trình.

### 2. Đăng ký khuôn mặt mới (Registration)
Sử dụng script đăng ký để thêm người dùng vào cơ sở dữ liệu:
```bash
python scripts/registration.py
```
- Nhập tên (Label) khi được hỏi.
- Di chuyển khuôn mặt vào khung vàng trên màn hình.
- Hệ thống sẽ tự động chụp 10 ảnh đạt chuẩn.

## Model
Các model sử dụng: https://drive.google.com/drive/folders/12AzwgwdY71oFnJRYaGC_VR9nFyZEyPrk?usp=sharing
(Lưu các file tải về vào thư mục `models/` nếu chưa có)
