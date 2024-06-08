from collections import defaultdict

import cv2, json, random
import numpy as np

from ultralytics import YOLO

def output_fn(prediction_output):
    infer = {}
    keypoints = {}
    boxes = {}

    for result in prediction_output:
        if not result.boxes == None:
            box = result.boxes
            if not box.cls == None:
                boxes['cls'] = box.cls.numpy().data.tolist()
            if not box.conf == None:
                boxes['conf'] = box.conf.numpy().data.tolist()
            if not box.data == None:
                boxes['data'] = box.data.numpy().data.tolist()
            if not box.id == None:
                boxes['id'] = box.id.numpy().data.tolist()
            if not box.xywh == None:
                boxes['xywh'] = box.xywh.numpy().data.tolist()
            if not box.xywhn == None:
                boxes['xywhn'] = box.xywhn.numpy().data.tolist()
            if not box.xyxy == None:
                boxes['xyxy'] = box.xyxy.numpy().data.tolist()
            if not box.xyxyn == None:
                boxes['xyxyn'] = box.xyxyn.numpy().data.tolist()

            infer['boxes'] = boxes
        if not result.keypoints == None:
            key = result.keypoints

            if not key.conf == None:
                keypoints['conf'] = key.conf.numpy().data.tolist()
            if not key.xy == None:
                keypoints['xy'] = key.xy.numpy().data.tolist()
            if not key.xyn == None:
                keypoints['xyn'] = key.xyn.numpy().data.tolist()

            infer['keypoints'] = keypoints
        if not result.masks == None:
            infer['masks'] = result.masks.numpy().data.tolist()
        if not result.probs == None:
            infer['probs'] = result.probs.numpy().data.tolist()
        if not result.names == None:
            infer['names'] = result.names
        if not result.speed == None:
            infer['speed'] = result.speed
    

    return infer

drawLines = [
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (11, 5),
    (12, 6),
    (11, 12),
    (13, 11),
    (13, 15),
    (14, 12),
    (14, 16),
]

model = YOLO('/Users/iminjun/Documents/git/yolo/models/yolov8n-pose.pt')

video_path = "/Users/iminjun/Documents/git/yolo/assests/video1.mp4"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('test.mp4', fourcc, 10, (width, height))

i = 0
while cap.isOpened():
    success, frame = cap.read()
    i += 1
    if success:
        if not (i % 3 == 0):
            continue        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        print(i)
        results = model.track(frame, persist=True)

        infer = output_fn(results)
        keypoints = infer['keypoints']


        for p in keypoints['xy']:
                if not p:
                    continue
                
                for dot in drawLines:
                    color = (0, 0, 0)
                
                    a, b = dot
                    if p[a] == [0, 0] or p[b] == [0, 0]:
                        continue

                    if a <= 2:
                        color = (255, 0, 0)
                    elif a <= 4:
                        color = (0, 255, 0)
                    elif a <= 8:
                        color = (0, 255, 255)
                    elif a <= 12:
                        color = (0, 255, 0)
                    elif a <= 16:
                        color = (0, 0, 255)
                        

                    cv2.line(frame, tuple(map(int, p[a])), tuple(map(int, p[b])), color, thickness=3)

                for kpt in p:
                    x, y = int(kpt[0]), int(kpt[1])
                    cv2.circle(frame, (x, y), radius=3, color=(0, 0, 0), thickness=5)

        # cv2.imshow("YOLOv8 Tracking", frame)

        out.write(frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

