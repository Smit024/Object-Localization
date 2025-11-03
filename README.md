Object Localization Using Traditional Machine Learning
End‑to‑end pipeline to detect and localize objects in 128×128 grayscale images with traditional ML and DL baselines. Includes preprocessing, model training, evaluation (MSE, IoU), and a Gradio demo.

Abstract
This project detects and localizes objects in grayscale images using traditional machine learning techniques. We built a complete ML pipeline from data preprocessing and EDA to training models and evaluating predictions using bounding box metrics like MSE and IoU.

Dataset
Images: 5,000 grayscale (128×128)

Annotations: CSV with columns x, y, width, height

Features: Flattened pixels (16,384 per image)

Folder layout expected:

data/images/ (PNG/JPG, 128×128, grayscale)

data/image_annotation.csv

Models Implemented
Linear Regression (baseline)

Random Forest (ensemble regression)

K‑Nearest Neighbors (instance‑based)

Multilayer Perceptron (feedforward NN)

Convolutional Neural Network (spatial features)

Results
Model	MSE	Avg IoU
Linear Regression	27.19	0.6681
Random Forest	28.84	0.6971
KNN	9.24	0.8461
MLP	21.49	0.7165
CNN	4.41	0.8457
Installation
bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
Usage
1) Train / Explore in Notebook
bash
jupyter notebook Group32_Object_Localization_ML_Final_Project.ipynb
Update any paths to data/images and data/image_annotation.csv if needed.

The notebook saves trained models into models/.

2) Run Gradio Demo
bash
python gradio_demo.py
Upload a 128×128 grayscale image and choose a model to visualize the predicted bounding box.

Project Structure
text
.
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ Group32_Object_Localization_ML_Final_Project.ipynb
├─ gradio_demo.py
├─ data/
│  ├─ images/                 # your 128x128 grayscale images
│  └─ image_annotation.csv    # x,y,width,height
├─ models/                    # saved models (excluded in .gitignore or use Git LFS)
│  ├─ linear_model.joblib
│  ├─ rf_model.joblib
│  ├─ knn_model.joblib
│  ├─ mlp_model.keras
│  └─ cnn_model.keras
└─ assets/                    # optional sample images, plots
Key Features
End‑to‑end pipeline: preprocessing, training, evaluation, visualization

Multiple models: traditional ML + DL baselines

Interactive demo: Gradio web UI for quick testing

Evaluation metrics: MSE and IoU for bounding boxes

Reproducibility
Set random seeds inside the notebook (NumPy, TensorFlow, scikit‑learn).

Document any non‑default hyperparameters in the notebook cells.

Future Work
PCA/autoencoders for dimensionality reduction

Grid/Bayesian hyperparameter tuning

CNN data augmentation

Multi‑object detection

Contributors
Smit Patel — data preprocessing, EDA, MLP/CNN implementation

Shubham Limbachiya — traditional ML models, hyperparameter tuning, documentation

License
This project is released under the MIT License. See LICENSE for details.

How to Cite
If you reference this work, please cite the repository:

text
@software{group32_object_localization_2025,
  title   = {Object Localization Using Traditional Machine Learning},
  author  = {Patel, Smit and Limbachiya, Shubham},
  year    = {2025},
  url     = {https://github.com/your-username/Object-Localization-ML}
}

