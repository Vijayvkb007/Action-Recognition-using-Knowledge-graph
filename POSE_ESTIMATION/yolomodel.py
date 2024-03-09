from ultralytics import YOLO
from matplotlib import pyplot as plt
import torch
import cv2

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

############################################
# YOLO MODEL

# load pretrained weights of model
model = YOLO('yolov8m-pose.pt')
model.to('cuda')

# video path file
video_path = r"/home/vijayvkb98/gitthing/knowledge-graph-for-action-understanding/DATASET/train_videos/0245.mp4"

# open video file
cap = cv2.VideoCapture(video_path)

# set the start and end frame indices
start_frame = 0
end_frame = 100

# Loop through frames
for frame_idx in range(start_frame, end_frame):
    ret, frame = cap.read()
    
    if not ret: break
        
    rframe = cv2.resize(frame, (224, 224))
    rframe = cv2.cvtColor(rframe, cv2.COLOR_BGR2RGB)
    
    results = model(source=rframe, conf=0.3, stream=True, device='cuda')
    
    for r in results:
        img = r.orig_img
        keypoints = r.keypoints.xy[0].cpu().numpy()
    cv2.imshow("frame",img)
		
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
		
cap.release()
cv2.destroyAllWindows()

