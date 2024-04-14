from ultralytics import YOLO
import numpy as np

classes = {
    0: ("personal_passport", 1),
    1: ("personal_passport", 2),
    2: ("vehicle_certificate", 1),
    3: ("vehicle_certificate", 2),
    4: ("vehicle_passport", 0),
    5: ("driver_license", 1),
    6: ("driver_license", 2)
}


def predict_class_proba_page_box(im):
    model = YOLO('../models/doc_detection.pt')
    results = model(im)
    prediction = results[0].boxes[0]

    proba = prediction.conf[0].item()
    pair = classes.get(int(prediction.cls))
    cls = pair[0]
    page = pair[1]
    box = np.array(prediction.xyxy)

    return cls, proba, page, box

def crop_box(im, box):
    x1, y1, x2, y2 = int(box[0][0]), int(box[0][1]), int(box[0][2]), int(box[0][3])
    crop_rectangle = (x1, y1, x2, y2)
    im = im.crop(crop_rectangle)
    return im