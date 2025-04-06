import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TRAIN_DIR = 'Waste Images\\train'
VAL_DIR = 'Waste Images\\validation'

class_weights = {0: 1., 1: 2.}

# Load training data
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='binary',
    image_size=IMG_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Load validation data
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels='inferred',
    label_mode='binary',
    image_size=IMG_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Base model (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze the base model layers

# Preprocessing layer for MobileNetV2
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Build the model
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = preprocess_input(inputs)  # Apply preprocessing
x = base_model(x, training=False)  # Run base model (inference mode)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)  # Optional dropout
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # Sigmoid for binary classification

model = tf.keras.Model(inputs, outputs)

model.summary()

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=validation_dataset,
    class_weight=class_weights,
    callbacks=[tensorboard_callback]
)

# Save the trained model
model.save('foodWasteClassifier2.keras')
print("Model saved successfully!")

# Check training and validation accuracy
print("Training accuracy:", history.history['accuracy'])
print("Validation accuracy:", history.history['val_accuracy'])

# Optionally, plot the training history to visualize accuracy and loss curves

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
