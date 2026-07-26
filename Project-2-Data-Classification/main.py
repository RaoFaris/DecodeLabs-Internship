"""
Project 2: Data Classification Using AI
DecodeLabs AI Internship

Goal: Build a basic classification model using a small dataset.
Dataset : Iris (sklearn built-in)
Algorithm: K-Nearest Neighbors (KNN)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ---------------------------------------------------------------
# Stage 2: Load the dataset
# ---------------------------------------------------------------
iris = load_iris()
X = iris.data                      # features (numeric measurements)
y = iris.target                    # target (species, encoded 0/1/2)
feature_names = iris.feature_names
target_names = iris.target_names

print("=" * 60)
print("STAGE 2: DATASET LOADED")
print("=" * 60)
print("Shape of X (samples, features):", X.shape)
print("Feature names:", feature_names)
print("Target names:", list(target_names))

# ---------------------------------------------------------------
# Stage 3: Explore the dataset
# ---------------------------------------------------------------
df = pd.DataFrame(X, columns=feature_names)
df["species"] = [target_names[i] for i in y]

print("\n" + "=" * 60)
print("STAGE 3: DATASET EXPLORATION")
print("=" * 60)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values per column:\n", df.isnull().sum())
print("\nStatistical summary:\n", df.describe())

# ---------------------------------------------------------------
# Stage 4: Separate Features (X) and Target (y)
# ---------------------------------------------------------------
# Already separated above as X and y when loading the dataset.

# ---------------------------------------------------------------
# Stage 6: Split into training and testing sets (BEFORE scaling)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print("\n" + "=" * 60)
print("STAGE 6: TRAIN/TEST SPLIT")
print("=" * 60)
print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])

# ---------------------------------------------------------------
# Stage 5: Scale the features (fit on TRAIN only, transform both)
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn mean/std from train
X_test_scaled = scaler.transform(X_test)         # apply same transform to test

print("\n" + "=" * 60)
print("STAGE 5: FEATURE SCALING")
print("=" * 60)
print("Mean of scaled training data (~0):", np.round(X_train_scaled.mean(axis=0), 3))
print("Std of scaled training data (~1):", np.round(X_train_scaled.std(axis=0), 3))

# ---------------------------------------------------------------
# Stage 7: Create the classification model
# ---------------------------------------------------------------
K = 5
model = KNeighborsClassifier(n_neighbors=K)

# ---------------------------------------------------------------
# Stage 8: Train the model
# ---------------------------------------------------------------
model.fit(X_train_scaled, y_train)

print("\n" + "=" * 60)
print("STAGE 8: MODEL TRAINED")
print("=" * 60)
print(f"KNN model trained with K={K} neighbors.")

# ---------------------------------------------------------------
# Stage 9: Make predictions
# ---------------------------------------------------------------
y_pred = model.predict(X_test_scaled)

# ---------------------------------------------------------------
# Stage 10: Evaluate the model
# ---------------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=target_names)

print("\n" + "=" * 60)
print("STAGE 10: MODEL EVALUATION")
print("=" * 60)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

# ---------------------------------------------------------------
# Stage 11: Test the model with completely new sample data
# ---------------------------------------------------------------
new_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],   # expected: setosa
    [6.7, 3.1, 4.7, 1.5],   # expected: versicolor
    [6.3, 3.3, 6.0, 2.5],   # expected: virginica
])
new_samples_scaled = scaler.transform(new_samples)   # use the SAME fitted scaler
new_predictions = model.predict(new_samples_scaled)

print("\n" + "=" * 60)
print("STAGE 11: PREDICTIONS ON NEW, UNSEEN DATA")
print("=" * 60)
for sample, pred in zip(new_samples, new_predictions):
    print(f"Input {sample} -> Predicted species: {target_names[pred]}")
