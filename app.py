import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt

model = load_model("model/brain_tumor_model.h5")


def predict_images():
    file_paths = filedialog.askopenfilenames()
    for widget in frame.winfo_children():
        widget.destroy()

    row = 0
    col = 0

    for file_path in file_paths:

        image = cv2.imread(file_path)
        image_resized = cv2.resize(image,(224,224))
        image_resized = image_resized/255.0
        image_resized = np.reshape(image_resized,(1,224,224,3))

        prediction = model.predict(image_resized)

        prob = prediction[0][0]

        if prob > 0.5:
            predicted = "Tumor"

        else:
            predicted = "No Tumor"
            

        # get actual label from folder
        actual = "Unknown"
        parent_folder = os.path.basename(os.path.dirname(file_path))

        if parent_folder == "yes":
            actual = "Tumor"
        elif parent_folder == "no":
            actual = "No Tumor"

        img = Image.open(file_path)
        img = img.resize((150,150))
        img = ImageTk.PhotoImage(img)

        panel = tk.Label(frame,image=img)
        panel.image = img
        panel.grid(row=row,column=col,padx=10,pady=10)

        # correctness check
        if actual != "Unknown":
            if predicted == actual:
                color = "green"
            else:
                color = "red"
        else:
            color = "black"

        text = tk.Label(
            frame,
            text=f"Predicted: {predicted} \nActual: {actual}",
            fg=color,
            font=("Arial",10,"bold")
        )

        text.grid(row=row+1,column=col)

        col += 1

        if col == 3:
            col = 0
            row += 2
window = tk.Tk()
window.title("Brain Tumor Detection Dashboard")
window.geometry("750x600")

btn = tk.Button(window,text="Select MRI Images",command=predict_images)
btn.pack(pady=20)

frame = tk.Frame(window)
frame.pack()

window.mainloop()
