import cv2

from ultralytics import YOLO

confidence_threshold     = 0.2
detected_above_threshold = False
model                    = YOLO("C:/Users/shapn/OneDrive/Desktop/Banana final python/src/train/leaf_classification_train/weights/best.pt")
class_names              = ['banana_leaf', 'leaf']

def detect_leaf_or_not(image):
    img     = cv2.imread(image)
    results = model(img)
    
    detected_above_threshold = False
    detected_classes         = []

    for box, cls in zip(results[0].boxes, results[0].boxes.cls):
        if float(box.conf) > confidence_threshold:
            detected_above_threshold = True
            detected_classes.append(class_names[int(cls)])  

    annotated_frame = results[0].plot()

    return annotated_frame, detected_classes

# Usage
# frame, classes = detect_leaf_disease("src/leaf_classification/infer_image/a.jpg")
# print(classes)
# cv2.imwrite("src/leaf_classification/output/Saved_image1.jpg", frame)