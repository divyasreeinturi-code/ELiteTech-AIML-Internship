
# AIML INTERNSHIP - TASK 3
# CAT AND DOG CLASSIFICATION USING SVM

# Step 1: Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import joblib


# Step 2: Dataset path
TRAIN_PATH = r"C:\Users\inturi divyasree\Downloads\archive (3)\train\train"

IMAGE_SIZE = (64, 64)
MAX_IMAGES = 500

images = []
labels = []

print("Loading training images...")


# Step 3: Load images and extract HOG features
for file in Path(TRAIN_PATH).rglob("*"):

    if not file.is_file():
        continue

    filename = file.name.lower()

    if filename.startswith("cat"):
        label = 0

    elif filename.startswith("dog"):
        label = 1

    else:
        continue

    if labels.count(label) >= MAX_IMAGES:
        continue

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
        labels.append(label)

    except Exception:
        continue


# Step 4: Display dataset information
print("Total images loaded:", len(images))
print("Cat images:", labels.count(0))
print("Dog images:", labels.count(1))


# Step 5: Convert into NumPy arrays
X = np.array(images)
y = np.array(labels)

if len(X) == 0 or len(np.unique(y)) != 2:
    raise ValueError(
        "No valid cat and dog images found. Check your folder and filenames."
    )


# Step 6: Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Step 7: Create and train SVM model
print("\nTraining SVM model...")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

model.fit(X_train, y_train)

print("SVM model trained successfully!")


# Step 8: Make predictions
y_pred = model.predict(X_test)


# Step 9: Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Cat", "Dog"]
    )
)


# Step 10: Display confusion matrix
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Cat", "Dog"]
)

display.plot(cmap="Blues")

plt.title("Cat and Dog Classification Using SVM")
plt.show()


# Step 11: Save trained model
joblib.dump(model, "Cat_Dog_SVM_Model.pkl")

print("\nModel saved successfully!")


# Step 12: Function to predict a new image
def predict_animal(image_path):

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

    result = "Cat" if prediction == 0 else "Dog"

    print("Predicted Animal:", result)

    plt.imshow(img_array, cmap="gray")
    plt.title("Prediction: " + result)
    plt.axis("off")
    plt.show()


# Step 13: Completion message
print("\nTask 3 Completed Successfully!")

