from ultralytics import YOLO
from matplotlib import pyplot as plt
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import numpy as np
import torch
import cv2

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def calculate_angle(a, b, c):
    a = np.array(a) # shoulder 
    b = np.array(b) # elbow
    c = np.array(c) # wrist

    # here 1 and 0 refers to y and x coord
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle





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
    rframe.flags.writeable = False
    
    results = model(source=rframe, conf=0.3, stream=True, device='cuda')
    
    rframe.flags.writeable = True
    
    for r in results:
        img = r.orig_img
        
        try:
            # obtain coordinates of keypoints
            keypoints = r.keypoints.xy[0].cpu().numpy()
            
            # use keypoints 5 for left shoulder , 7 for left elbow, 9 for left wrist
            shoulder, elbow, wrist = keypoints[5], keypoints[7], keypoints[9]
                    
            # get the angle
            angle = calculate_angle(shoulder, elbow, wrist)
            cv2.putText(img, f"Angle: {angle:.2f}", (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        except:
            pass
    cv2.imshow("frame",img)
		
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
		
cap.release()
cv2.destroyAllWindows()




# MEDIAPIPE MODEL

video_path = r"/home/vijayvkb98/gitthing/knowledge-graph-for-action-understanding/DATASET/train_videos/0245.mp4"

# open video file
cap = cv2.VideoCapture(video_path)
# plt.figure(figsize=(13,13))

# set the start and end frame indices
start_frame = 0
end_frame = 100

# convert to see video 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter("output.avi", fourcc, 30, (224, 224))

# set up mediapipe instanace
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Loop through frames
    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()

        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rframe = cv2.resize(frame, (224, 224))
        rframe.flags.writeable = False
        
        # make detection 
        results = pose.process(rframe)
        # print(results)
        rframe.flags.writeable = True
        
        # extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # get coordinates of the keypoints
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # get the angle
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # print the angle, and change the normalized one to actual coordinates and convert back into int
            # args for put text at last 2 for line width, 0.5 size, () color
            cv2.putText(rframe, f"Angle: {angle:.2f}", 
                        (10,15),
                       # tuple(np.multiply(elbow, [224, 224]).astype(int)),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,)
                        
            
        except:
            pass
        
        # render detection 
        mp_drawing.draw_landmarks(rframe, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(255, 117, 66), thickness=2, circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(245, 0, 0), thickness=2, circle_radius=2))
        
        
        video.write(rframe)

        cv2.imshow("frame",rframe)
            
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

cap.release()
plt.show()
