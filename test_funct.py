from utils.augmentations import letterbox
import cv2    
from efficientad import xywh2xyxy, adjust_boxes, parse_yolo_annotation
import numpy as np

# 定义function 绘制边框，输入img，box
def draw_boxes(img, boxes):
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img

source = '/home/wudidi/code/yolo-distillation-AD/datasets/hefei_yolo_format_v2.4/train/images/backpack-box_good_0.jpg'
img0 = cv2.imread(source)
print(img0.shape)
img_h, img_w, _ = img0.shape

img, ratio, (dw, dh) = letterbox(img0,  auto=False)
print(img.shape)
cv2.imwrite('letterbox.jpg', img)



txt_path = str(source).replace('images', 'labels').replace('.jpg', '.txt')
annotations = parse_yolo_annotation(txt_path)
for annotation in annotations:
    class_id, x, y, width, height = annotation
    x1, y1, x2, y2 = xywh2xyxy(x, y, width, height, img_w, img_h)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img0, str(class_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite('raw_with_box.jpg', img0)
    
    #对边框进行调整
    boxes = [[x1, y1, x2, y2]]
    adjust_box = adjust_boxes(boxes, ratio, dw, dh)
    x1, y1, x2, y2 = adjust_box[0]
    # 转换为整数坐标
    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, str(class_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite('letterbox_with_box.jpg', img)