import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_yolo_model():
    weights_path = "yolov3.weights"  # Path to YOLO weights file
    config_path = "yolov3.cfg"       # Path to YOLO config file
    labels_path = "coco.names"       # Path to file containing class names
    
    net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    labels = open(labels_path).read().strip().split("\n")
    
    return net, output_layers, labels

def detect_defects_yolo(net, output_layers, image):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)
    
    boxes = []
    confidences = []
    class_ids = []
    
    for detection in detections:
        for object_detection in detection:
            scores = object_detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(object_detection[0] * width)
                center_y = int(object_detection[1] * height)
                w = int(object_detection[2] * width)
                h = int(object_detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    return boxes, confidences, class_ids

def comparator(img_dir, design_img_name, product_img_name, net, output_layers):
    design_img_path = os.path.join(img_dir, design_img_name)
    product_img_path = os.path.join(img_dir, product_img_name)

    design_img = cv2.imread(design_img_path)
    product_img = cv2.imread(product_img_path)

    boxes, confidences, class_ids = detect_defects_yolo(net, output_layers, product_img)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(product_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    result_path = os.path.join(img_dir, f"{product_img_name.split('.')[0]}_result.bmp")
    cv2.imwrite(result_path, product_img)
    print(f"Saved result to {result_path}")

image_dir = "./dfscer43t/photoset1"
net, output_layers, labels = load_yolo_model()

# for i in range(1, 25):
for i in range(17, 18):
    comparator(image_dir, f"{i:02d}_design.bmp", f"{i:02d}.bmp", net, output_layers)
    print(f"Processed image {i:02d}")
