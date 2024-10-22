import os
import cv2
from ultralytics import YOLO

# Load YOLOv8 models
bottle_detector = YOLO('yolov8n.pt', task='detect')
bacardi_classifier = YOLO('testing/models/bacardi_cls_oct21.pt', task='classify')

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))          
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Step 1: Detect bottles
        results = bottle_detector(frame)
        
        # Iterate over detections
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            label = bottle_detector.names[class_id]

            if label == 'bottle':
                # Crop the detected bottle
                cropped_bottle = frame[y1:y2, x1:x2]
                
                try:
                    # Step 2: Classify the cropped bottle
                    classify_result = bacardi_classifier(cropped_bottle)[0]
                    
                    # Get classification results
                    if hasattr(classify_result, 'probs'):
                        # Get the class with highest probability
                        class_idx = int(classify_result.probs.top1)
                        confidence = float(classify_result.probs.top1conf)
                        class_label = classify_result.names[class_idx]
                    else:
                        class_label = "Unknown"
                        confidence = 0.0
                    
                    # Step 3: Annotate the frame
                    label_text = f"{class_label} ({confidence:.2f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label_text, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Display the frame (optional)
                    cv2.imshow('Processing', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                except Exception as e:
                    print(f"Error in classification: {str(e)}")
                    continue
        
        out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    for video_file in video_files:
        print(f"Processing: {video_file}")
        video_path = os.path.join(input_folder, video_file)  # Fixed here: video_file instead of video_path
        output_path = os.path.join(output_folder, f"processed_{video_file}")
        process_video(video_path, output_path)

# Example usage
input_folder = "testing/videos"
output_folder = "testing/results"
process_folder(input_folder, output_folder)