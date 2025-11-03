import gradio as gr
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image, ImageDraw
import os

# --- Load All Models ---
# This script expects the model files to be in the same directory.
try:
    reg_lin = joblib.load("linear_model.joblib")
    reg_rf = joblib.load("rf_model.joblib")
    reg_knn = joblib.load("knn_model.joblib")
    mlp_model = tf.keras.models.load_model("mlp_model.keras")
    cnn_model = tf.keras.models.load_model("cnn_model.keras")
    print("All models loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load all models. Error: {e}")
    print("The app will run, but model choices may fail.")
    print("Please download and unzip 'object_localization_models.zip' into this directory.")
    reg_lin, reg_rf, reg_knn, mlp_model, cnn_model = None, None, None, None, None

# --- Prediction Function ---
def predict_bbox(image, model_choice):
    """
    Takes a PIL image and model choice, returns a PIL image with bounding box.
    """
    if not model_choice:
        return image, "Please select a model from the dropdown."
        
    if not image:
        return None, "Please upload an image."

    # Preprocess the image
    image_gray = image.convert("L").resize((128, 128))
    img_array = np.array(image_gray)
    img_draw = image_gray.convert("RGB") # Create a copy for drawing

    pred_box = [10, 10, 40, 40] # Default box if model fails
    info = f"Model: {model_choice}\n"

    try:
        if model_choice in ["Linear Regression", "Random Forest", "KNN", "MLP"]:
            # Flatten image for these models
            img_flat = img_array.flatten().reshape(1, -1) / 255.0

            if model_choice == "Linear Regression" and reg_lin:
                pred_box = reg_lin.predict(img_flat)[0]
            elif model_choice == "Random Forest" and reg_rf:
                pred_box = reg_rf.predict(img_flat)[0]
            elif model_choice == "KNN" and reg_knn:
                pred_box = reg_knn.predict(img_flat)[0]
            elif model_choice == "MLP" and mlp_model:
                pred_box = mlp_model.predict(img_flat)[0]
            else:
                raise FileNotFoundError(f"{model_choice} model is not loaded.")

        elif model_choice == "CNN" and cnn_model:
            # Reshape for CNN
            img_reshaped = img_array.reshape(1, 128, 128, 1) / 255.0
            pred_box = cnn_model.predict(img_reshaped)[0]
        else:
             raise FileNotFoundError(f"{model_choice} model is not loaded.")

        # Draw the predicted bounding box
        draw = ImageDraw.Draw(img_draw)
        x, y, w, h = pred_box
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
        
        info += f"Box: [x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}]"
        return img_draw, info

    except Exception as e:
        return image_gray.convert("RGB"), f"Error: {e}"

# --- Gradio Interface ---
interface = gr.Interface(
    fn=predict_bbox,
    inputs=[
        gr.Image(type="pil", label="Upload Grayscale Image (128x128)"),
        gr.Dropdown(
            choices=["Linear Regression", "Random Forest", "KNN", "MLP", "CNN"],
            label="Select Model"
        )
    ],
    outputs=[
        gr.Image(label="Predicted Bounding Box"),
        gr.Text(label="Prediction Info")
    ],
    title="Object Localization Demo – Group 32",
    description="Upload a grayscale image and select a model to see its bounding box prediction. (Based on CSCI 4750 Final Project by Smit Patel & Shubham Limbachiya)",
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch()
