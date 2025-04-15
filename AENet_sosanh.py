import torch
import onnxruntime as ort
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from PIL import Image
import time
import torchvision.transforms as transforms  # Thêm dòng này để import transforms

# Load model PyTorch (TorchScript)
def load_pytorch_model(model_path):
    model = torch.jit.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

# Load model NCNN (ONNX)
def load_onnx_model(onnx_model_path):
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    return session, input_name

# Preprocessing for PyTorch model
def preprocess_pytorch_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0)
    return img

# Preprocessing for NCNN model
def preprocess_onnx_frame(frame, session, input_name):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0).numpy()
    return session.run(None, {input_name: img})[0].flatten()

# Function to predict using PyTorch model
def predict_pytorch_model(model, frame):
    img_tensor = preprocess_pytorch_frame(frame)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1).numpy()[0]
        return probabilities[0], probabilities[1]

# Function to predict using NCNN model
def predict_onnx_model(session, input_name, frame):
    embedding = preprocess_onnx_frame(frame, session, input_name)
    return embedding

# Compare models for performance and prediction accuracy
def compare_models(model_pytorch, model_onnx, frame):
    # Time for PyTorch model prediction
    start_pytorch = time.time()
    prob_live_pytorch, prob_spoof_pytorch = predict_pytorch_model(model_pytorch, frame)
    time_pytorch = time.time() - start_pytorch
    
    # Time for NCNN model prediction
    start_onnx = time.time()
    embedding_onnx = predict_onnx_model(model_onnx, input_name, frame)
    time_onnx = time.time() - start_onnx
    
    # Calculate cosine similarity between PyTorch predictions and ONNX embedding
    # You can choose to compute cosine similarity between PyTorch's 'probabilities' and 'embedding_onnx' if needed
    similarity = cosine([prob_live_pytorch, prob_spoof_pytorch], embedding_onnx[:2])
    
    # Print results
    print(f"PyTorch model: Live: {prob_live_pytorch:.2f}, Spoof: {prob_spoof_pytorch:.2f} (Time: {time_pytorch:.4f}s)")
    print(f"ONNX model: Time: {time_onnx:.4f}s")
    print(f"Cosine Similarity between PyTorch and ONNX predictions: {similarity:.4f}")
    
    return prob_live_pytorch, prob_spoof_pytorch, embedding_onnx, time_pytorch, time_onnx, similarity

# Main function to load image and compare models
def compare_with_image(model_pytorch, model_onnx, input_name, image_path):
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print("Không thể đọc ảnh. Kiểm tra lại đường dẫn ảnh.")
        return

    # Run prediction for both models
    prob_live_pytorch, prob_spoof_pytorch, embedding_onnx, time_pytorch, time_onnx, similarity = compare_models(model_pytorch, model_onnx, frame)

    # Show predictions on image
    label = "Live" if prob_live_pytorch > 0.95 else "Spoof"
    cv2.putText(frame, f"Live: {prob_live_pytorch:.2f} | Spoof: {prob_spoof_pytorch:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Cosine Similarity: {similarity:.4f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Face Anti-Spoofing", frame)
    cv2.waitKey(0)  # Wait for a key press to close the image window
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Load both models
    model_pytorch = load_pytorch_model("AENet.pt")
    model_onnx, input_name = load_onnx_model("w600k_r50.onnx")
    
    # Specify the image path
    image_path = "dataset/train/Akshay Kumar/anh_10.jpg"
    
    # Compare models with a single image
    compare_with_image(model_pytorch, model_onnx, input_name, image_path)
