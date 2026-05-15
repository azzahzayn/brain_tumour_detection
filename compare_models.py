import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2


# ---------------- LOAD DATA ----------------
data = []
labels = []

categories = ["no", "yes"]

for category in categories:
    path = os.path.join("dataset", category)
    label = categories.index(category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)

        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))

        data.append(image)
        labels.append(label)

# normalize
data = np.array(data).astype("float32") / 255.0
labels = np.array(labels)

print("Images loaded:", len(data))


# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42
)


# ---------------- CNN MODEL ----------------
cnn_model = Sequential()

cnn_model.add(Input(shape=(224, 224, 3)))

cnn_model.add(Conv2D(32, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(2, 2))

cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(2, 2))

cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(2, 2))

cnn_model.add(Flatten())

cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dropout(0.5))

cnn_model.add(Dense(1, activation='sigmoid'))

cnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nTraining CNN...\n")

history_cnn = cnn_model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_test, y_test)
)


# ---------------- MobileNetV2 MODEL ----------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# freeze base layers
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

mobilenet_model = Model(
    inputs=base_model.input,
    outputs=output
)

mobilenet_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nTraining MobileNetV2...\n")

history_mobilenet = mobilenet_model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_test, y_test)
)


# ---------------- ACCURACY COMPARISON ----------------
cnn_acc = history_cnn.history['val_accuracy'][-1]
mobilenet_acc = history_mobilenet.history['val_accuracy'][-1]

print("\nModel Comparison:")
print(f"CNN Accuracy: {cnn_acc:.4f}")
print(f"MobileNetV2 Accuracy: {mobilenet_acc:.4f}")


# ---------------- GRAPH ----------------
plt.figure()

plt.plot(history_cnn.history['val_accuracy'], label='CNN')
plt.plot(history_mobilenet.history['val_accuracy'], label='MobileNetV2')

plt.title("Model Comparison (Validation Accuracy)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()