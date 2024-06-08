import torch, json, cv2
import numpy as np

from ultralytics import YOLO


def model_fn(model_dir):
    print("Executing model_fn from inference.py ...")
    model = YOLO('/opt/ml/model/yolov8n-pose.pt')

    print('read model!')
    return model

def input_fn(request_body, request_content_type):
    print("Executing input_fn from inference.py ...")
    arr = bytes(request_body)

    encoded_img = np.frombuffer(arr, dtype=np.uint8)
    img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

    return img
    
def predict_fn(input_data, model):
    print("Executing predict_fn from inference.py ...")
    result = model.track(input_data, persist=True)

    print("predicted!")
    print(result)
    return result
        
def output_fn(prediction_output, content_type):
    print("Executing output_fn from inference.py ...")

    infer = {}
    keypoints = {}
    boxes = {}

    for result in prediction_output:
        print(result)
        if not result.boxes == None:
            box = result.boxes
            if not box.cls == None:
                boxes['cls'] = box.cls.numpy().data.tolist()
            if not box.conf == None:
                boxes['conf'] = box.conf.numpy().data.tolist()
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
    
    print('infer:', infer)
    return json.dumps(infer)