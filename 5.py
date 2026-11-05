
import os
import numpy as np
import joblib

from PIL import Image
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------------
# 1. DATASET PATH
# -----------------------------------

BASE_PATH = r"C:\Users\inturi divyasree\Downloads\archive (5)"

TRAIN_PATH = os.path.join(BASE_PATH, "training")
EVAL_PATH = os.path.join(BASE_PATH, "evaluation")

IMAGE_SIZE = (64, 64)

CLASSES = [
    "Bread",
    "Dairy product",
    "Dessert"
]

# Approximate calories per 100 grams
CALORIES = {
    "Bread": 265,
    "Dairy product": 150,
    "Dessert": 350
}

# -----------------------------------
# 2. IMAGE FEATURE EXTRACTION
# -----------------------------------

def extract_features(image_path):

    try:
        image = Image.open(image_path).convert("L")
        image = image.resize(IMAGE_SIZE)

        image_array = np.array(image)

        features = hog(
            image_array,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )

        return features

    except Exception as e:
        print("Error:", image_path, e)
        return None


# -----------------------------------
# 3. LOAD DATASET
# -----------------------------------

def load_dataset(folder_path):

    X = []
    y = []

    for class_name in CLASSES:

        class_path = os.path.join(folder_path, class_name)

        if not os.path.isdir(class_path):
            print("Missing folder:", class_name)
            continue

        count = 0

        for filename in os.listdir(class_path):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            ):

                image_path = os.path.join(class_path, filename)

                features = extract_features(image_path)

                if features is not None:
                    X.append(features)
                    y.append(class_name)
                    count += 1

        print(class_name, "images loaded:", count)

    return np.array(X), np.array(y)


# -----------------------------------
# 4. LOAD TRAINING IMAGES
# -----------------------------------

print("\nLoading training dataset...")

X_train, y_train = load_dataset(TRAIN_PATH)

if len(X_train) == 0:
    raise ValueError("No training images found. Check dataset path.")

print("\nTotal training images:", len(X_train))

# -----------------------------------
# 5. SPLIT TRAINING DATA
# -----------------------------------

X1, X2, y1, y2 = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

print("\nTraining samples:", len(X1))
print("Testing samples:", len(X2))

# -----------------------------------
# 6. TRAIN SVM MODEL
# -----------------------------------

print("\nTraining SVM model...")

validation_model = LinearSVC(
    C=1.0,
    max_iter=10000,
    random_state=42
)

validation_model.fit(X1, y1)

# -----------------------------------
# 7. VALIDATION RESULTS
# -----------------------------------

predictions = validation_model.predict(X2)

accuracy = accuracy_score(y2, predictions)

print("\nValidation Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y2,
    predictions,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y2,
    predictions,
    labels=CLASSES
))

# -----------------------------------
# 8. TRAIN FINAL MODEL
# -----------------------------------

print("\nTraining final model using all training images...")

model = LinearSVC(
    C=1.0,
    max_iter=10000,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------------
# 9. EVALUATE ON EVALUATION FOLDER
# -----------------------------------

print("\nLoading evaluation images...")

X_eval, y_eval = load_dataset(EVAL_PATH)

if len(X_eval) > 0:

    eval_predictions = model.predict(X_eval)

    eval_accuracy = accuracy_score(
        y_eval,
        eval_predictions
    )

    print("\nEvaluation Accuracy:",
          round(eval_accuracy * 100, 2), "%")

    print("\nEvaluation Classification Report:")
    print(classification_report(
        y_eval,
        eval_predictions,
        labels=CLASSES,
        zero_division=0
    ))

else:
    print("No matching evaluation images found.")

# -----------------------------------
# 10. SAVE MODEL
# -----------------------------------

joblib.dump(
    {
        "model": model,
        "classes": CLASSES,
        "calories": CALORIES
    },
    "Food_Recognition_Model.pkl"
)

print("\nModel saved successfully!")

# -----------------------------------
# 11. FOOD PREDICTION
# -----------------------------------

def predict_food(image_path, weight_grams=100):

    features = extract_features(image_path)

    if features is None:
        print("Unable to read image.")
        return

    features = features.reshape(1, -1)

    predicted_food = model.predict(features)[0]

    calories_per_100g = CALORIES[predicted_food]

    estimated_calories = (
        calories_per_100g * weight_grams / 100
    )

    print("\n========== FOOD PREDICTION ==========")

    print("Predicted Food:", predicted_food)

    print("Food Weight:", weight_grams, "grams")

    print("Calories per 100g:", calories_per_100g, "kcal")

    print("Estimated Calories:",
          round(estimated_calories, 2), "kcal")

    print("=====================================")


# -----------------------------------
# 12. TEST YOUR OWN FOOD IMAGE
# -----------------------------------

print("\nEnter the image path to predict food.")

image_path = input("Image path: ").strip().strip('"')

if os.path.isfile(image_path):

    weight = float(input("Enter food weight in grams: "))

    predict_food(image_path, weight)

else:
    print("Image file not found. Model training is complete.")
