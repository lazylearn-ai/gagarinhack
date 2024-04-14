from ultralytics import YOLO
import numpy as np
from math import atan, pi
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd =  '/usr/bin/tesseract'


ratios = {
    "personal_passport": 4,
    "vehicle_certificate": 4,
    "driver_license": 4,
    "vehicle_passport": 4
}


nums = "1234567890"

def rotate_image(im):
    w, h = im.size
    angle = atan(h / w) * 180 / pi 
    size = (max(w, h), max(w, h))
    img_w, img_h = im.size
    im1 = Image.new('RGB', size, (255, 255, 255))
    bg_w, bg_h = im1.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    im1.paste(im, offset)

    im2 = Image.new('RGB', size, (255, 255, 255))
    bg_w, bg_h = im2.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    im2.paste(im, offset)
     
    im1 = im1.rotate(angle)
    im2 = im2.rotate(180+angle)

    return im1, im2

def predict_text(im):
    model = YOLO('../models/number_detection.pt')

    results = model(im)
    prediction = results[0].boxes
    coord = np.array(prediction.xyxy)
    x1, y1, x2, y2 = int(coord[0][0]), int(coord[0][1]), int(coord[0][2]), int(coord[0][3])
    crop_rectangle = (x1, y1, x2, y2)
    im = im.crop(crop_rectangle)
    ret = ""
    count = 1
    while len(ret) < 7:
        text = pytesseract.image_to_string(im)
        if len(text) > 0:
            for i in text:
                if type(i) == type("str"):
                    ret += i
        ret = "".join(ret.split())
        print(ret)
        if len(ret) > 7 or count == 3:
            break
        else:
            im = im.rotate(90)
            count += 1
            ret = ""
    return ret

def split_to_seria_and_number(text, cls):
    if len(text) == 0:
        return "", ""
    else:
        ratio = ratios[cls]
        seria = text[:ratio]
        number = text[ratio:]
        return seria, number
