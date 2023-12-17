import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from ultralytics import YOLO

# #2.2 
# ##Step 2: YOLOv8 Training
# ### Training Model 
# model = YOLO('yolov8n.pt')  # build a new model from YAML
# results = model.train(data='C:/Users/Owner/Documents/Python/AER850P3/Project 3 Data/data/data.yaml', epochs=100, imgsz=900, batch = 5)


#2.3 
##YOLOv8 Evaluation
### Testing Model 

def modelpred(name, path):
    results = model.predict(path)
    annotated_frame = results[0].plot( pil=True, font_size=60)  # Adjust the text_size as needed
    file_name = name + '.jpg'
    cv2.imwrite(file_name, annotated_frame)


model = YOLO('runs/detect/train3/weights/best.pt')
ardmega_img = cv2.imread('Project 3 Data/data/evaluation/ardmega.jpg')
arduno_img = cv2.imread('Project 3 Data/data/evaluation/arduno.jpg')
rasppi_img = cv2.imread('Project 3 Data/data/evaluation/rasppi.jpg')

modelpred('ardmega_img',ardmega_img)
modelpred('arduno_img',arduno_img)
modelpred('rasppi_img',rasppi_img)

