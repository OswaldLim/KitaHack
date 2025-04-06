import cv2
import numpy as np
import tensorflow.lite as tflite
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from ultralytics import YOLO
import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (1 = warning, 2 = error, 3 = ignore)
warnings.filterwarnings("ignore")  # Suppress other warnings

yolo_model = YOLO('yolov8n.pt')

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="foodWasteClassifier2.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess images
def preprocess_image(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  # Resize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Detect objects with YOLO
    results = yolo_model(frame)[0]  # YOLOv8 detection result

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0]
        if conf < 0.5:
            continue  # Skip low confidence

        # Step 2: Crop the detected region
        cropped = frame[y1:y2, x1:x2]

        # Step 3: Run through your classifier
        input_img = preprocess_image(cropped).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_img)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])
        label = "Other Waste" if pred[0] > 0.5 else "Food Waste"


        # Step 4: Draw results
        color = (0, 255, 0) if label == "Food Waste" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


    # cv2.imshow("Cropped",cropped)
    cv2.imshow("YOLO + Classifier", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()