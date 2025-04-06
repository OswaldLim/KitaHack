import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the trained model
model = tf.keras.models.load_model("foodWasteClassifier2.keras")

test_dir = "Test Data"

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='binary',  # Ensure this matches your training mode
    image_size=(224, 224),
    batch_size=32
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_dataset)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Load and preprocess a specific image (bananaTest.jpg)
img = cv2.imread("Test Data\\Other_waste\\otherTest.jpg")
if img is None:
    print("Error: Image not found.")
else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    resize = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)

    plt.imshow(resize)  # Display the first image in the batch
    plt.title("Preprocessed Image")
    plt.axis('off')
    plt.show()
    
    input_image = np.expand_dims(resize, axis=0)  # Add batch dimension



    # Predict with the model
    yhat = model.predict(input_image)

    # Print prediction for the specific image
    print(f"Raw prediction for bananaTest.jpg: {yhat}, Shape: {yhat.shape}")

    # Interpret the prediction
    predicted_label = 1 if yhat[0][0] > 0.5 else 0
    print(f"Predicted class: {predicted_label}")