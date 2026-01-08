import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

def align_face_by_eyes(image, left_eye, right_eye):
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return aligned_face

def resize_and_pad(image, target_size=(112, 112)):
    """
    Resizes the image maintaining aspect ratio and pads with black pixels.
    Returns the preprocessed tensor ready for ArcFace (N, C, H, W).
    """
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    delta_w = target_size[1] - new_w
    delta_h = target_size[0] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded

def preprocess_for_arcface(image):
    """
    Normalizes and converts to channel-first format.
    """
    # Assuming image is already resized and padded (112, 112)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_transposed = np.transpose(rgb_img, (2, 0, 1))
    img_normalized = (img_transposed - 127.5) / 127.5
    img_expanded = np.expand_dims(img_normalized, axis=0).astype(np.float32)
    return img_expanded

def preprocess_for_liveness(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0)
    return img
