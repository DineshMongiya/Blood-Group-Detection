import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from blood_detection.ml.preprocess import preprocess_image
from blood_detection.ml.model import create_model

# Dataset path and label mapping
dataset_path = "./dataset"
label_mapping = {
    'A+': 0, 'A-': 1, 'B+': 2, 'B-': 3,
    'AB+': 4, 'AB-': 5, 'O+': 6, 'O-': 7
}

# Step 1: Load and preprocess data
X, y = [], []
for label, value in label_mapping.items():
    folder_path = os.path.join(dataset_path, label)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder for blood group {label} not found: {folder_path}")
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        X.append(preprocess_image(img_path))  # Preprocess the image
        y.append(value)  # Assign the label

# Convert to NumPy arrays
X = np.array(X).reshape(-1, 128, 128, 1) / 255.0  # Normalize images
y = np.array(y)

# Shuffle the dataset
X, y = shuffle(X, y, random_state=42)

# Step 2: Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Data augmentation for training data
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

# No augmentation for validation data
validation_datagen = ImageDataGenerator()
validation_generator = validation_datagen.flow(X_val, y_val, batch_size=32)

# Step 4: Create and compile the model
model = create_model()

# Step 5: Add callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Step 6: Train the model
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=[early_stopping, lr_scheduler]
)

# Step 7: Save the trained model
os.makedirs('./blood_detection/ml', exist_ok=True)
model.save('./blood_detection/ml/model.keras')
print("Model saved successfully!")
