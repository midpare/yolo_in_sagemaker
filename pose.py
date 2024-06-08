import cv2, json, random, math
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

possessed = {}

model1 = YOLO('/Users/iminjun/Documents/git/yolo/models/yolov8n-pose.pt')
model2 = YOLO('/Users/iminjun/Documents/git/yolo/models/yolov8s.pt')

video_path = "/Users/iminjun/Documents/git/yolo/assests/video4.mp4"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_video.mp4', fourcc, 10, (width, height))
f = 0

while cap.isOpened():
    success, frame = cap.read()

    if success:
        f += 1
        if not (f % 3 == 0):
            continue

        if f > 240:
            break

        print(f)
        # print(json.dumps(infer['keypoints'], sort_keys=True, indent=4))
        result1 = model1.track(frame, persist=True)

        infer1 = output_fn(result1)
        keypoints = infer1['keypoints']

        # print(json.dumps(keypoints, sort_keys=True, indent=4))
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

        peopleId = [4]

        results2 = model2.track(frame, persist=True)

        infer2 = output_fn(results2)


        if 'boxes' in infer2 and len(infer2['boxes']['cls']) > 0:
            boxes = infer2['boxes']

            
            for i in range(len(boxes['cls'])):
                x1, y1, x2, y2 = boxes['xyxy'][i]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = boxes['conf'][i]
                cls = boxes['cls'][i]
                id = '0'
                
                if 'id' in boxes and boxes['id']:
                    id = boxes['id'][i]
                
                if id in peopleId and f > 60:
                    for j in range(len(boxes['cls'])):
                        print(boxes['id'])

                        _x1, _y1, _x2, _y2 = boxes['xyxy'][j]
                        _x1, _y1, _x2, _y2 = int(_x1), int(_y1), int(_x2), int(_y2)

                        middle_x, middle_y = (_x1 + _x2) / 2, (_y1 + _y2) / 2

                        rh_x, rh_y = keypoints['xy'][0][9]
                        lh_x, lh_y = keypoints['xy'][0][10]

                        distance = math.sqrt(((middle_x - lh_x) ** 2) + ((middle_y - lh_y) ** 2))

                        print('id:', boxes['id'][j], 'distance:', distance)
                        if not boxes['id'][j] == id and math.sqrt((middle_x - lh_x) ** 2 + (middle_y - lh_y) ** 2) < 100:
                            print('id:', boxes['id'][j])
                            
                            object_id = boxes['id'][j]
                            if not possessed.get(id):
                                possessed[id] = []

                            if not object_id in possessed[id]:
                                possessed[id].append(object_id)


                print(possessed)
                for i in possessed:
                    for object in possessed[i]:
                        index = boxes['id'].index(object)
                        _x1, _y1, _x2, _y2 = boxes['xyxy'][index]
                        _x1, _y1, _x2, _y2 = int(_x1), int(_y1), int(_x2), int(_y2)
                        cv2.putText(frame, f"possessed by Id:{int(i)}", (_x1,_y1-70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        if f > 165:
                            cv2.putText(frame, f"Warning!", (_x1,_y1-100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)




                color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 0, 0), 4)
                cv2.putText(frame, f"id: {int(id)}", (x1,y1-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Class: {infer2['names'][int(cls)]}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # print(peopleId)
            # for i in range(len(boxes['cls'])):
        

        cv2.imshow("YOLOv8 Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        out.write(frame)
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()