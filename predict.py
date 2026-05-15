import cv2
import numpy as np
from tensorflow.keras.models import load_model

# load trained model
model = load_model("model/brain_tumor_model.h5")

# load test image
image = cv2.imread("test_images/test.jpg")
image = cv2.resize(image,(224,224))
image = image/255.0
image = np.reshape(image,(1,224,224,3))

prediction = model.predict(image)

if prediction[0][0] > 0.5:
    print("Tumor Detected")
else:
    print("No Tumor")