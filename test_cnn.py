import os
import cv2  # Used for image reading and processing
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Define Constants ---
MODEL_PATH = 'traffic_sign_model.keras'  # <--- Your model file path
TEST_DATA_DIR = 'dataset_split/test'  # <--- *** Modify this to your test dataset root directory ***

# !! Important: The image dimensions and number of classes here must be exactly the same as used during training !!
IMG_HEIGHT = 32  # Image height used during training
IMG_WIDTH = 32   # Image width used during training
# Theoretical total number of classes, even if the test set does not contain all classes, the model output dimension is still this
NUM_CLASSES = 58  # Total number of classes during training (0 to 57)

# --- 2. Function to Load Data (similar to the training script) ---
# Note: Ensure that the preprocessing performed by this function (especially resize) is exactly the same as during training
def load_data(data_dir, img_height, img_width, num_classes):
    """
    Loads image data and labels from the specified directory.

    Args:
        data_dir (str): Path to the root directory of the dataset.
        img_height (int): Target image height.
        img_width (int): Target image width.
        num_classes (int): Theoretical total number of classes (used for folder names).

    Returns:
        tuple: Contains two NumPy arrays (images, labels) or (None, None).
               images: Image data with shape (number of samples, img_height, img_width, 3).
               labels: Label data with shape (number of samples,) (integers from 0 to num_classes-1).
    """
    images = []
    labels = []
    print(f"Starting to load test data from '{data_dir}'...")
    if not os.path.isdir(data_dir):
        print(f"Error: Test data directory '{data_dir}' does not exist. Please check the TEST_DATA_DIR variable.")
        return None, None

    # Iterate through folders from 0 to num_classes-1
    for label in range(num_classes):
        class_dir = os.path.join(data_dir, str(label))
        if not os.path.isdir(class_dir):
            # It's normal for the test set to lack folders for certain classes, print a message and continue
            # print(f"Info: Test class folder '{class_dir}' not found, skipping.")
            continue

        # print(f"  Loading test images for class {label}...") # Can uncomment to see details
        loaded_count = 0
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm', '.bmp')):
                img_path = os.path.join(class_dir, filename)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"    Warning: Could not read image '{img_path}', skipping.")
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img, (img_width, img_height))
                    images.append(img_resized)
                    labels.append(label)  # Use the original 0-57 labels
                    loaded_count += 1
                except Exception as e:
                    print(f"    Error processing image '{img_path}': {e}")
        # if loaded_count > 0:
            # print(f"    Successfully loaded {loaded_count} test images.")

    if not images:
        print("Error: Failed to load any images from the test directory. Please check the dataset structure and file formats.")
        return None, None

    print(f"Test data loading complete. Loaded a total of {len(images)} images.")
    return np.array(images), np.array(labels)

# --- 3. Load Model ---
print(f"Loading model: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    exit()

try:
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    model.summary()  # Print model structure to confirm correctness
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 4. Load and Preprocess Test Data ---
X_test_raw, y_test = load_data(TEST_DATA_DIR, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES)

if X_test_raw is not None and y_test is not None:
    print(f"Test image data shape: {X_test_raw.shape}")
    print(f"Test label data shape: {y_test.shape}")

    # Normalization (must be consistent with training)
    X_test = X_test_raw.astype('float32') / 255.0

    # --- 5. Evaluate Model ---
    print("\nEvaluating model on the test set...")
    # Use model.evaluate to get overall loss and accuracy
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest set loss (Loss): {loss:.4f}")
    print(f"Test set accuracy (Accuracy): {accuracy:.4f}")

    # --- 6. Make Predictions and Generate Detailed Report ---
    print("\nGenerating prediction results and detailed evaluation report...")
    # Get the model's prediction probabilities for the test set
    y_pred_probs = model.predict(X_test, verbose=1)
    # Convert probabilities to the most likely class labels (0-57)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # Calculate and print the classification report (Precision, Recall, F1-score)
    # Get all the actual class labels present in the test data
    present_classes = sorted(np.unique(y_test))
    target_names = [str(i) for i in present_classes]  # Generate names only for the existing classes

    # Note: If some classes have no samples in the test set (not present in y_test),
    # classification_report will handle this automatically or give a warning.
    # We can explicitly specify to report only these classes using labels=present_classes.
    print("\nClassification Report:")
    try:
        report = classification_report(y_test, y_pred_classes, labels=present_classes, target_names=target_names, zero_division=0)
        print(report)
    except Exception as e:
        print(f"Error generating classification report: {e}")
        # If the above fails, try without specifying labels and target_names
        try:
            report = classification_report(y_test, y_pred_classes, zero_division=0)
            print(report)
        except Exception as e2:
             print(f"Failed to generate classification report again: {e2}")


    # --- 7. (Optional) Calculate and Visualize Confusion Matrix ---
    print("\nGenerating confusion matrix...")
    try:
        cm = confusion_matrix(y_test, y_pred_classes, labels=present_classes)  # Calculate only based on existing classes

        plt.figure(figsize=(15, 12))  # May need to adjust size to fit the number of classes
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix on Test Set')
        plt.tight_layout()
        # plt.savefig('confusion_matrix.png') # Can optionally save the image
        plt.show()
    except Exception as e:
        print(f"Error generating or displaying confusion matrix: {e}")

    # --- 8. (Optional) Display Some Test Samples and Their Predictions ---
    num_samples_to_show = 15
    indices = np.random.choice(range(len(X_test)), num_samples_to_show, replace=False)

    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indices):
        plt.subplot(3, 5, i + 1)  # Adjust layout to match num_samples_to_show
        plt.imshow(X_test_raw[idx])  # Display the original unnormalized image
        plt.axis('off')
        predicted_label = y_pred_classes[idx]
        true_label = y_test[idx]
        color = 'green' if predicted_label == true_label else 'red'
        plt.title(f"Pred: {predicted_label}, True: {true_label}", color=color)

    plt.suptitle("Sample Test Images with Predictions")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
    plt.show()

else:
    print("Model evaluation cannot be performed because test data loading failed.")