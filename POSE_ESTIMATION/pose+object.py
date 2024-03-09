from ultralytics import YOLO
from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch

# use cuda if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# load model
# model = YOLO("yolov8x-pose-p6.pt").to(device)
model = YOLO("yolov8l-pose.pt").to(device)

def calculate_angle(a, b, c):
    """
    Calculate the angle between b point
    or angle between the line ab and bc
    
    input:
        a, b, c: coordinates of the keypoints/joints
        
    output:
        angle: angle between the line ab and bc
    """
    
    a = np.array(a) # eg: shoulder
    b = np.array(b) # eg: elbow
    c = np.array(c) # eg: wrist
    
    # here 0 and 1 refers to the x and y coordinates
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def verb_detection(keypoints,):
    """
    Detect the verb from the keypoints
    
    input:
        keypoints: list of keypoints
        
    output:
        verb: detected verb
    """
    # use keypoints 5 for left shoulder , 7 for left elbow, 9 for left wrist
    ls, le, lw = keypoints[5], keypoints[7], keypoints[9]
    rs, re, rw = keypoints[6], keypoints[8], keypoints[10]
    
    # get the angle
    angle1 = calculate_angle(ls, le, lw)
    angle2 = calculate_angle(rs, re, rw)
    
    if ls[0] - 10 <= rw[0] <= ls[0] + 10:
        verb = "right swing" 
        print(verb)
        T = True
    else:
        verb = "swing back"  
        print(verb)
    if rs[0] - 10 <= lw[0] <= rs[0] + 10:
        print("left swing")
        
    return verb

def mark_keypoints(rframe, keypoints):
    """
    Mark the keypoints on the image
    
    input:
        keypoints: list of keypoints
        
    output:
        img: image with keypoints marked
    """
    # use keypoints 5 for left shoulder , 7 for left elbow, 9 for left wrist
    shoulder, elbow, wrist = keypoints[5], keypoints[7], keypoints[9]
    shoulder1, elbow1, wrist1 = keypoints[6], keypoints[8], keypoints[10]                
    l11, l21, l31 = keypoints[11], keypoints[13], keypoints[15] 
    l12, l22, l32 = keypoints[12], keypoints[14], keypoints[16] 
    # get the angle
    angle = calculate_angle(shoulder, elbow, wrist)
    cv2.putText(rframe, f"angle: {angle:.2f}", (int(elbow[0]) + 10, int(elbow[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(rframe, (int(shoulder[0]), int(shoulder[1])), 5, (0, 0, 255), -1)
    cv2.circle(rframe, (int(elbow[0]), int(elbow[1])), 5, (0, 0, 255), -1)
    cv2.circle(rframe, (int(wrist[0]), int(wrist[1])), 5, (0, 0, 255), -1)
    cv2.line(rframe, (int(shoulder[0]), int(shoulder[1])), (int(elbow[0]), int(elbow[1])), (0, 255, 0), 2)
    cv2.line(rframe, (int(elbow[0]), int(elbow[1])),(int(wrist[0]), int(wrist[1])) , (0, 255, 0), 2)
    
    # get the angle
    angle = calculate_angle(shoulder1, elbow1, wrist1)
    cv2.putText(rframe, f"angle: {angle:.2f}", (int(elbow1[0]) + 10, int(elbow1[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(rframe, (int(shoulder1[0]), int(shoulder1[1])), 5, (0, 0, 255), -1)
    cv2.circle(rframe, (int(elbow1[0]), int(elbow1[1])), 5, (0, 0, 255), -1)
    cv2.circle(rframe, (int(wrist1[0]), int(wrist1[1])), 5, (0, 0, 255), -1)
    cv2.line(rframe, (int(shoulder1[0]), int(shoulder1[1])), (int(elbow1[0]), int(elbow1[1])), (0, 255, 0), 2)
    cv2.line(rframe, (int(elbow1[0]), int(elbow1[1])),(int(wrist1[0]), int(wrist1[1])) , (0, 255, 0), 2)
    
    angle = calculate_angle(l12, l22, l32)
    cv2.putText(rframe, f"angle: {angle:.2f}", (int(l22[0]) + 10, int(l22[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(rframe, (int(l12[0]), int(l12[1])), 5, (0, 0, 255), -1)
    cv2.circle(rframe, (int(l22[0]), int(l22[1])), 5, (0, 0, 255), -1)
    cv2.circle(rframe, (int(l32[0]), int(l32[1])), 5, (0, 0, 255), -1)
    cv2.line(rframe, (int(l12[0]), int(l12[1])), (int(l22[0]), int(l22[1])), (0, 255, 0), 2)
    cv2.line(rframe, (int(l22[0]), int(l22[1])),(int(l32[0]), int(l32[1])) , (0, 255, 0), 2)
    
    angle = calculate_angle(l11, l21, l31)
    cv2.putText(rframe, f"angle: {angle:.2f}", (int(l21[0]) + 10, int(l21[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(rframe, (int(l11[0]), int(l11[1])), 5, (0, 0, 255), -1)
    cv2.circle(rframe, (int(l21[0]), int(l21[1])), 5, (0, 0, 255), -1)
    cv2.circle(rframe, (int(l31[0]), int(l31[1])), 5, (0, 0, 255), -1)
    cv2.line(rframe, (int(l11[0]), int(l11[1])), (int(l21[0]), int(l21[1])), (0, 255, 0), 2)
    cv2.line(rframe, (int(l21[0]), int(l21[1])),(int(l31[0]), int(l31[1])) , (0, 255, 0), 2)
    
    cv2.putText(rframe, f"{verb_detection(keypoints)}", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# video path file
video_path = r"/home/vijayvkb98/gitthing/knowledge-graph-for-action-understanding/CKT_DATASET/Bowled/001.mp4"

# open video file
cap = cv2.VideoCapture(video_path)

# set the start and end frame indices
start_frame = 0
end_frame = 100
T = False
# Loop through frames
for frame_idx in range(start_frame, end_frame):
    ret, frame = cap.read()
    
    if not ret: break
        
    rframe = cv2.resize(frame, (640, 640))
    # rframe = cv2.resize(frame, (224, 224))
    # rframe = cv2.cvtColor(rframe, cv2.COLOR_BGR2RGB)
    rframe.flags.writeable = False
    
    results = model(source=rframe, conf=0.3, stream=True, device='cuda')
    
    rframe.flags.writeable = True
    
    for r in results:
        img = r.orig_img
        
        try:
            # obtain coordinates of keypoints
            keypoints = r.keypoints.xy[0].cpu().numpy()
            mark_keypoints(rframe, keypoints)
        except:
            pass
		
    cv2.imshow("frame",rframe)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
		
cap.release()
cv2.destroyAllWindows()

model = YOLO("yolov8x.pt").to(device)

# video path file
video_path = r"/home/vijayvkb98/gitthing/knowledge-graph-for-action-understanding/CKT_DATASET/Bowled/001.mp4"

# open video file
cap = cv2.VideoCapture(video_path)

# set the start and end frame indices
start_frame = 0
end_frame = 100

# Loop through frames
# for frame_idx in range(start_frame, end_frame):
while True:
    ret, frame = cap.read()
    
    if not ret: break
        
    rframe = cv2.resize(frame, (640, 640))    
    results = model(source=rframe, conf=0.6, stream=True, device='cuda')

    for r in results:
        img = r.orig_img
        boxes = r.boxes
        # print(len(boxes))
        for box in boxes:
            # print(f"{box[0]}")
            x, y, x1, y1 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)
            cv2.putText(rframe, f"person", (x+10,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0,0), 2)
            cv2.imshow("frame",img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
