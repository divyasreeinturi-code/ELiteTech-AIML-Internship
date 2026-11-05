
# AIML INTERNSHIP - TASK 4
# HAND GESTURE RECOGNITION USING SVM

# Step 1: Import libraries

import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import joblib


# Step 2: Dataset path

DATASET_PATH = r"C:\Users\inturi divyasree\Downloads\archive\leapGestRecog"

IMAGE_SIZE = (64, 64)
MAX_IMAGES_PER_CLASS = 300

# Gesture folder names

GESTURES = [
    "01_palm",
    "02_l",
    "03_fist",
    "04_fist_moved",
    "05_thumb",
    "06_index",
    "07_ok",
    "08_palm_moved",
    "09_c",
    "10_down"
]

# Step 3: Initialize variables

images = []
labels = []

print("Loading hand gesture images...")


# Step 4: Load images and extract HOG features

for gesture in GESTURES:

    count = 0

    for file in Path(DATASET_PATH).rglob("*"):

        if not file.is_file():
            continue

        if file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        # Find the matching gesture folder
        if gesture.lower() not in [
            part.lower() for part in file.parts
        ]:
            continue

        if count >= MAX_IMAGES_PER_CLASS:
            break

        try:

            img = Image.open(file).convert("L")
            img = img.resize(IMAGE_SIZE)

            img_array = np.array(img)

            features = hog(
                img_array,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys"
            )

            images.append(features)
            labels.append(gesture)

            count += 1

        except Exception:
            continue

    print(gesture, "images loaded:", count)


# Step 5: Check dataset

print("\nTotal images loaded:", len(images))

if len(images) == 0 or len(set(labels)) < 2:
    raise ValueError(
        "No valid gesture images found. Check dataset folder structure."
    )


# Step 6: Convert into NumPy arrays

X = np.array(images)
y = np.array(labels)

print("Feature shape:", X.shape)


# Step 7: Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Step 8: Train SVM classifier

print("\nTraining SVM model...")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

model.fit(X_train, y_train)

print("SVM model trained successfully!")


# Step 9: Predict gestures

y_pred = model.predict(X_test)


# Step 10: Evaluate model

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:",
      round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=GESTURES,
        zero_division=0
    )
)


# Step 11: Confusion matrix

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=GESTURES
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=GESTURES
)

display.plot(
    cmap="Blues",
    xticks_rotation=45
)

plt.title("Hand Gesture Recognition Using SVM")
plt.tight_layout()
plt.show()


# Step 12: Save trained model

model_data = {
    "model": model,
    "gestures": GESTURES,
    "image_size": IMAGE_SIZE
}

joblib.dump(
    model_data,
    "Hand_Gesture_SVM_Model.pkl"
)

print("\nModel saved successfully!")


# Step 13: Predict a new hand gesture image

def predict_gesture(image_path):

    img = Image.open(image_path).convert("L")
    img = img.resize(IMAGE_SIZE)

    img_array = np.array(img)

    features = hog(
        img_array,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    prediction = model.predict([features])[0]

    print("\nPredicted Hand Gesture:", prediction)

    plt.imshow(img_array, cmap="gray")
    plt.title("Predicted Gesture: " + prediction)
    plt.axis("off")
    plt.show()


# Step 14: Completion message

print("\nHand Gesture Recognition Task Completed Successfully!")
