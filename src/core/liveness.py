import torch
import numpy as np
import onnxruntime as ort
from src.utils.image_utils import preprocess_for_liveness

class LivenessDetector:
    def __init__(self, model_path, providers=['CPUExecutionProvider']):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, frame):
        """
        Returns (label, prob_live, prob_spoof)
        """
        img_tensor = preprocess_for_liveness(frame)
        onnx_input = img_tensor.numpy().astype(np.float32)
        onnx_output = self.session.run(None, {self.input_name: onnx_input})[0]
        probabilities = torch.nn.functional.softmax(torch.tensor(onnx_output), dim=1).numpy()[0]
        
        # Original logic: label = "Spoof" if probabilities[1] > probabilities[0] else "Live"
        label = "Spoof" if probabilities[1] > probabilities[0] else "Live"
        return label, probabilities[0], probabilities[1]
