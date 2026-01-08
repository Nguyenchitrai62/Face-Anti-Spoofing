import time
from ultralytics import YOLO
import numpy as np

class FaceDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def detect(self, img, conf_threshold=0.7):
        start_time = time.time()
        # verbose=False to suppress stdout
        results = self.model.predict(source=img, conf=conf_threshold, verbose=False)
        elapsed = time.time() - start_time
        
        detections = results[0].boxes
        keypoints_list = results[0].keypoints.xy if results[0].keypoints is not None else None
        
        results_list = []
        for i, box in enumerate(detections):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            
            keypoints = []
            if keypoints_list is not None and i < len(keypoints_list):
                 kps = keypoints_list[i].cpu().numpy().astype(int)
                 keypoints = [(int(x), int(y)) for x, y in kps]
            
            results_list.append({
                'xyxy': [x1, y1, x2, y2],
                'conf': conf,
                'keypoints': keypoints
            })
            
        return results_list, elapsed
