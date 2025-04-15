import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

model = load_model("traffic_sign_model.keras")

img_path = "dataset_split/test/25/025_0002.png"
img = image.load_img(img_path, target_size=(32, 32))  # resize
img_array = image.img_to_array(img) / 255.0            # normalize
img_array = np.expand_dims(img_array, axis=0)          # batch (1, 32, 32, 3)

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
confidence = np.max(predictions)

print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")

plt.imshow(img)
plt.title(f"Predicted: {predicted_class} ({confidence:.2f})")
plt.axis('off')
plt.show()
