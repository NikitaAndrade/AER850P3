import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from ultralytics import YOLO

# #2.1 
# ##step 1: Object Masking 
# ###Reading Image
# image_path = 'motherboard_image.JPEG'
# image = cv2.imread(image_path)

# ###Image Scaling
# target_width, target_height = int(2172/2.5),int(2896/2.5)
# resized_image = cv2.resize(image, (target_width, target_height))
# rotated_image = cv2.rotate(resized_image, cv2.ROTATE_90_CLOCKWISE)

# ###Applying Image Masks
# blurred_image = cv2.GaussianBlur(rotated_image, (5, 5), 0)
# gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
# threshold_value = 110
# _, binary_thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

# ###Inverting Masks to Detect Object, Extracting Contours 
# inverted_mask = cv2.bitwise_not(binary_thresholded_image)
# contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# image_with_contours = rotated_image.copy()
# largest_contour = max(contours, key=cv2.contourArea)
# mask = np.zeros_like(gray_image)
# cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
# image_with_contours = rotated_image.copy()
# cv2.drawContours(image_with_contours, [largest_contour], -1, (0, 255, 0), 2)
# pcb_extracted = cv2.bitwise_and(rotated_image, rotated_image, mask=mask)

# ###Dispaly and Save Images 
# cv2.imshow('Extracted PCB', pcb_extracted)
# cv2.imshow('Inverted Mask', inverted_mask)
# cv2.imshow('Image with Largest Contour', image_with_contours)
# cv2.imwrite('pcb_extracted.jpg', pcb_extracted)
# cv2.imwrite('edge_detection.jpg', image_with_contours)
# cv2.imwrite('mask_detetction.jpg', inverted_mask)

# cv2.waitKey(0)  
# cv2.destroyAllWindows() 

