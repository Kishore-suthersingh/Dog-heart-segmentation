from ultralytics import YOLO

import cv2
model = YOLO("Enter the path of your model")  


image_path = "Enter the image path"

img = cv2.imread(image_path)
H, W, _ = img.shape

result = model(img)

for result in result:
    for j, mask in enumerate(result.masks.data):

        mask = mask.numpy() * 255

        mask = cv2.resize(mask, (W, H))

        cv2.imwrite('./output.png', mask)