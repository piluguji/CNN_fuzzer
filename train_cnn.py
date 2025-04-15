import os
import cv2  # for image reading and processing (install: pip install opencv-python)
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- 1. Define Constants ---
DATA_DIR = 'dataset_split/train'  # <--- *** Modify this to your dataset root directory ***
IMG_HEIGHT = 32  # Adjust to your desired image height (common for traffic sign datasets)
IMG_WIDTH = 32   # Adjust to your desired image width
NUM_CLASSES = 58 # Your number of classes (0 to 57)
EPOCHS = 20      # Total number of training epochs (adjust as needed)
BATCH_SIZE = 32  # Number of samples per training batch (adjust as needed)
VALIDATION_SPLIT = 0.2 # Proportion of data to use for validation

# --- 2. Load Data ---
def load_data(data_dir):
    """
    Loads image data and labels from the specified directory.

    Args:
        data_dir (str): Path to the dataset root directory, containing subfolders named 0-57.

    Returns:
        tuple: Contains two NumPy arrays (images, labels).
               images: Image data with shape (number of samples, IMG_HEIGHT, IMG_WIDTH, 3).
               labels: Label data with shape (number of samples,) (integers from 0-57).
    """
    images = []
    labels = []
    print(f"Starting to load data from '{data_dir}'...")
    if not os.path.isdir(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist. Please check the DATA_DIR variable.")
        return None, None

    # Iterate through folders 0 to 57
    for label in range(NUM_CLASSES):
        class_dir = os.path.join(data_dir, str(label))
        if not os.path.isdir(class_dir):
            print(f"Warning: Class folder '{class_dir}' not found, skipping.")
            continue

        print(f"  Loading images for class {label}...")
        loaded_count = 0
        for filename in os.listdir(class_dir):
            # Only process common image file extensions
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm', '.bmp')):
                img_path = os.path.join(class_dir, filename)
                try:
                    # Read image (OpenCV uses BGR format by default)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"    Warning: Could not read image '{img_path}', skipping.")
                        continue
                    # Convert image from BGR to RGB (Keras models typically expect RGB)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Resize the image
                    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    images.append(img_resized)
                    labels.append(label)
                    loaded_count += 1
                except Exception as e:
                    print(f"    Error processing image '{img_path}': {e}")
        print(f"    Successfully loaded {loaded_count} images.")
        min_samples_per_class = 2
        if 0 < loaded_count < min_samples_per_class:
            print(f"    Warning: Class {label} only has {loaded_count} samples, duplicating samples to meet stratification requirements.")
            # Find the index of the first sample of this class just added
            start_index = len(images) - loaded_count
            # Number of duplicates needed
            num_duplicates_needed = min_samples_per_class - loaded_count
            for i in range(num_duplicates_needed):
                # Duplicate the first image (or the last, the effect is the same)
                images.append(images[start_index])
                labels.append(labels[start_index])
            print(f"    Increased the number of samples for class {label} to {min_samples_per_class}.")

    if not images:
        print("Error: Failed to load any images. Please check the dataset structure and file formats.")
        return None, None

    print(f"Data loading complete. Total of {len(images)} images loaded.")
    # Convert lists to NumPy arrays
    return np.array(images), np.array(labels)

# --- 3. Data Preprocessing ---
images, labels = load_data(DATA_DIR)

if images is not None and labels is not None:
    print(f"Image data shape: {images.shape}")  # Should be (number of samples, height, width, 3)
    print(f"Label data shape: {labels.shape}")    # Should be (number of samples,)

    # Normalize pixel values to the range [0, 1]
    images = images.astype('float32') / 255.0

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels,
        test_size=VALIDATION_SPLIT,
        random_state=42,  # For reproducible results
        stratify=labels   # Ensure similar class distribution in training and validation sets
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

    # --- 4. Build CNN Model ---
    def build_model(input_shape, num_classes):
        """Builds the CNN model architecture"""
        model = models.Sequential()

        # Convolutional layer 1
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Convolutional layer 2
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Convolutional layer 3
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Flatten layer
        model.add(layers.Flatten())

        # Fully connected layer 1
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5)) # Add Dropout to prevent overfitting

        # Output layer
        model.add(layers.Dense(num_classes, activation='softmax')) # Use softmax for multi-class classification

        return model

    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3) # Input shape (height, width, number of color channels)
    model = build_model(input_shape, NUM_CLASSES)

    # Print model summary
    model.summary()

    # --- 5. Compile Model ---
    model.compile(optimizer='adam', # Adam optimizer usually works well
                  loss='sparse_categorical_crossentropy', # Use this loss function because labels are integers (0-57)
                  metrics=['accuracy']) # Evaluation metric is accuracy

    # --- 6. Train Model ---
    print("\nStarting model training...")
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(X_val, y_val))
    print("Model training complete.")

    # --- 7. Evaluate Model ---
    print("\nEvaluating model on the validation set...")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=2)
    print(f"Validation set loss: {val_loss:.4f}")
    print(f"Validation set accuracy: {val_acc:.4f}")

    # --- 8. (Optional) Save Model ---
    try:
        # Recommended to use Keras v3's .keras format
        model.save('traffic_sign_model.keras')
        print("\nModel saved as 'traffic_sign_model.keras'")
        # Or use HDF5 format (.h5)
        # model.save('traffic_sign_model.h5')
        # print("\nModel saved as 'traffic_sign_model.h5'")
    except Exception as e:
        print(f"\nError saving the model: {e}")


    # --- 9. (Optional) Visualize Training Process ---
    def plot_history(history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(len(acc)) # Use the actual number of training epochs

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.tight_layout()
        plt.show()

    plot_history(history)

else:
    print("Model training cannot proceed due to data loading failure.")