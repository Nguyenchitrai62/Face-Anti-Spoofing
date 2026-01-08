import numpy as np
import onnxruntime as ort
import faiss
from scipy.spatial.distance import cosine
from src.utils.image_utils import resize_and_pad, preprocess_for_arcface

class FaceRecognizer:
    def __init__(self, model_path, index_path, labels_path, providers=['CPUExecutionProvider']):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
        self.index = None
        self.labels = None
        try:
            # Check if files exist
            import os
            if os.path.exists(index_path) and os.path.exists(labels_path):
                self.index = faiss.read_index(str(index_path))
                self.labels = np.load(str(labels_path))
            else:
                print(f"Warning: Index or Labels file not found at {index_path} / {labels_path}")
        except Exception as e:
            print(f"Warning: Could not load index or labels: {e}")
            
    def get_embedding(self, face_image):
        """
        Takes a face crop (BGR), preprocesses it, and runs inference.
        Returns the 512-d embedding.
        """
        padded = resize_and_pad(face_image)
        input_tensor = preprocess_for_arcface(padded)
        
        embedding = self.session.run(None, {self.input_name: input_tensor})[0]
        return embedding.flatten()

    def identify_face(self, embedding, threshold=0.7):
        if self.index is None or self.labels is None:
            return "unknown", 0.0
            
        score_max = -1
        query_vector = embedding.reshape(-1)
        
        # Naive linear scan using FAISS storage (as per original code)
        for i in range(self.index.ntotal):
            stored_vector = np.zeros((512,), dtype=np.float32)
            self.index.reconstruct(i, stored_vector)
            similarity_score = 1 - cosine(query_vector, stored_vector)
            
            if similarity_score > threshold:
                return self.labels[i], similarity_score
            
            if similarity_score > score_max:
                score_max = similarity_score
                
        return "unknown", score_max
