import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

model = tf.keras.models.load_model("foodWasteClassifier2.keras")

# Convert to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open("foodWasteClassifier2.tflite", "wb") as f:
    f.write(tflite_model)

print("Model successfully converted to TensorFlow Lite!")

