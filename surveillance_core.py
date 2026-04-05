import cv2
import numpy as np
import time
from ultralytics import YOLO
import math

class BorderSurveillanceSystem:
    def __init__(self):
        # Load YOLOv8 nano model for fast real-time performance
        try:
            self.model = YOLO('yolov8n.pt')
        except Exception as e:
            print("Failed to load YOLO model:", e)
        
        # Dictionary to store tracking state: {track_id: {'first_seen': time, 'last_pos': (x,y), 'history': [(x,y)]}}
        self.track_history = {}
        
        # Define a virtual border line (x-coords, this is vertical for simplicity: x=320 if w=640)
        self.BORDER_X = 320 
        
        # Loitering threshold (seconds)
        self.LOITER_THRESHOLD = 5

    def get_color(self, label):
        if label == 'person':
            return (0, 0, 255) # Red for humans
        elif label in ['car', 'truck', 'motorcycle']:
            return (0, 165, 255) # Orange for vehicles
        elif label in ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow']:
            return (0, 255, 0) # Green for animals
        return (255, 0, 0) # Blue for others

    def process_frame(self, frame):
        """
        Process a single video frame. Returns the annotated frame and list of threats.
        """
        if frame is None:
            return None, []
            
        h, w, _ = frame.shape
        self.BORDER_X = int(w / 2) # Set border in the middle of frame
        
        results = self.model.track(frame, persist=True, verbose=False)
        annotated_frame = frame.copy()
        
        # Draw the virtual border line
        cv2.line(annotated_frame, (self.BORDER_X, 0), (self.BORDER_X, h), (0, 255, 255), 2)
        cv2.putText(annotated_frame, "VIRTUAL BORDER", (self.BORDER_X + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        threats = []
        current_time = time.time()
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, cls_id, conf in zip(boxes, track_ids, classes, confidences):
                label = self.model.names[cls_id]
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # Check tracking history
                if track_id not in self.track_history:
                    self.track_history[track_id] = {
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'last_pos': (cx, cy),
                        'history': [(cx, cy)],
                        'crossed': False
                    }
                
                state = self.track_history[track_id]
                state['history'].append((cx, cy))
                if len(state['history']) > 30: # keep last 30 points
                    state['history'].pop(0)
                
                time_in_zone = current_time - state['first_seen']
                
                # Intrusion logic:
                is_threat = False
                alert_text = label
                color = self.get_color(label)
                
                # 1. Line Crossing (Vector intersection approx - if previous x was one side, current is other)
                prev_cx = state['last_pos'][0]
                if (prev_cx < self.BORDER_X and cx >= self.BORDER_X) or (prev_cx > self.BORDER_X and cx <= self.BORDER_X):
                    state['crossed'] = True
                
                if state['crossed'] and label in ['person', 'car', 'truck', 'motorcycle']:
                    is_threat = True
                    alert_text = f"BORDER CROSSED: {label}"
                    color = (0, 0, 255) # Strict red
                    
                # 2. Loitering
                if time_in_zone > self.LOITER_THRESHOLD and not state['crossed']:
                    # Calculate if they moved significantly
                    start_pos = state['history'][0]
                    dist = math.dist(start_pos, (cx, cy))
                    if dist < 50 and label == 'person': # Stayed roughly in same area
                        is_threat = True
                        alert_text = f"LOITERING: {label}"
                        color = (0, 165, 255) # Orange Warning
                
                state['last_pos'] = (cx, cy)
                state['last_seen'] = current_time
                
                # Draw Box and Track
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f"ID:{track_id} {alert_text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw trail
                points = np.array(state['history'], np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=color, thickness=2)
                
                if is_threat:
                    threats.append({
                        'id': track_id,
                        'class': label,
                        'reason': alert_text.split(':')[0],
                        'time_in_zone': round(time_in_zone, 1)
                    })
                    
        # Cleanup old tracks
        tracks_to_delete = [tid for tid, state in self.track_history.items() if current_time - state['last_seen'] > 5.0]
        for tid in tracks_to_delete:
            del self.track_history[tid]
            
        return annotated_frame, threats
