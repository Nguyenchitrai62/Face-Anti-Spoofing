import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = os.path.join(BASE_DIR, "models")

# Model Files
ARCFACE_MODEL_NAME = "w600k_r50.onnx"
YOLO_MODEL_NAME = "yolov8n-face.pt"
AENET_MODEL_NAME = "AENet.onnx"

ARCFACE_MODEL_PATH = os.path.join(MODELS_DIR, ARCFACE_MODEL_NAME)
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, YOLO_MODEL_NAME)
AENET_MODEL_PATH = os.path.join(MODELS_DIR, AENET_MODEL_NAME)


DATA_DIR = os.path.join(BASE_DIR, "data")

FACE_INDEX_PATH = os.path.join(DATA_DIR, "face_index.bin")
FACE_LABELS_PATH = os.path.join(DATA_DIR, "face_labels.npy")

DEFAULT_CONFIDENCE = 0.7
