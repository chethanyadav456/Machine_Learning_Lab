"""
Case Study 4: Image Classification of Animals

Scenario:
A zoo wants to automate the process of identifying animals captured by surveillance
cameras. By classifying images into categories such as lions, elephants, and zebras,
the zoo can monitor animal activity more efficiently and detect any anomalies
automatically.

Objective:
Build a Neural Network model to classify animal images into different categories using
a deep learning framework like TensorFlow/Keras.

Concept Used:
Supervised Learning with a Convolutional Neural Network (CNN) — a type of deep neural
network designed for image recognition tasks. CNNs automatically extract image features
like edges, textures, and shapes to classify objects efficiently.
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load and preprocess dataset (CIFAR-10)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to range [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Class names in CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10,
                     validation_data=(test_images, test_labels))

# Evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest Accuracy: {test_acc:.2f}")
