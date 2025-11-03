# End-to-End Object Localization with ML/DL

This repository contains the final project for **CSCI 4750 – Machine Learning**, developed by **Group 32 (Smit Patel, Shubham Limbachiya)**. The project implements a full pipeline to detect and localize objects in grayscale images by predicting bounding box coordinates.

We compare the performance of traditional machine learning models (Linear Regression, Random Forest, KNN) against deep learning approaches (MLP, CNN) to determine the most effective method for this regression-based image task.

## Live Demo

An interactive web demo was built with Gradio to allow live predictions. You can upload any 128x128 grayscale image and select one of the five trained models to see its predicted bounding box.

*(To add a screenshot, run `app.py` locally, take a screenshot, upload it to your GitHub repo, and replace the line below)*
`![Gradio App Demo](httpsDELETEME/path/to/your/demo_screenshot.png)`

## Model Performance

Models were evaluated on **Mean Squared Error (MSE)** for coordinate accuracy and **Intersection over Union (IoU)** for spatial overlap. The **CNN** and **KNN** models performed the best, achieving the highest IoU scores.

| Model | MSE | Average IoU |
| :--- | :---: | :---: |
| Linear Regression | 27.19 | 0.6681 |
| Random Forest | 28.84 | 0.6971 |
| MLP (Keras) | 21.49 | 0.7165 |
| **CNN (Keras)** | **4.41** | **0.8457** |
| **KNN (n=3)** | **9.24** | **0.8461** |

### Visual Comparison

The difference in performance is clear when visualizing the predictions. KNN and CNN (not shown) capture the object's location accurately, while Linear Regression and Random Forest struggle with the spatial complexity.

*(You can get these images by running the visualization cells [7cAWBtygTto4], [eaSqE-hdUKlb], and [HHxocUa1UX83] in your notebook, saving the plots, and uploading them to a `visualizations` folder in your repo.)*

| KNN Prediction (High IoU) | Random Forest Prediction | Linear Regression (Low IoU) |
| :---: | :---: | :---: |
| ![KNN Example](./visualizations/knn_vs_actual.png) | ![RF Example](./visualizations/rf_vs_actual.png) | ![LR Example](./visualizations/lr_vs_actual.png) |

---

## Installation & Setup

**1. Clone the Repository**
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
```

**2. Create and Activate a Virtual Environment**
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

**3. Install Dependencies**
This project requires several Python libraries. A `requirements.txt` file is provided for easy installation.
```bash
pip install -r requirements.txt
```

**4. Download Data and Models**

Due to their size, the image dataset (`images_repo.zip`) and the trained models (`.joblib`, `.keras`) are not stored in this GitHub repository.

* **Data:** Download the dataset from [**<-- PASTE YOUR GOOGLE DRIVE/DROPBOX LINK HERE**]. Unzip and place the `images_repo` folder in the root of this project.
* **Models:** Download the trained models from [**<-- PASTE YOUR GOOGLE DRIVE/DROPBOX LINK HERE**]. Unzip and place all `.joblib` and `.keras` files in the root of this project.

---

## How to Run

### 1. Run the Jupyter Notebook
To see the full process of data loading, EDA, model training, and evaluation, you can run the main notebook:
```bash
jupyter notebook Group32_Object_Localization_ML_Final_Project.ipynb
```

### 2. Launch the Gradio Web Demo
To run the interactive web demo, simply execute the `app.py` script. This requires all model files (`.joblib`, `.keras`) to be in the same directory.
```bash
python app.py
```
Open the local URL (e.g., `http://127.0.0.1:7860`) in your browser.

---

## Future Work

To enhance this project, future steps could include:
-   Applying **dimensionality reduction** (PCA, Autoencoders) to speed up training for traditional models.
-   Performing **rigorous hyperparameter tuning** using GridSearchCV or Bayesian Optimization.
-   Adding **data augmentation** to improve the CNN model's generalization.
-   Extending the dataset and models to handle **multi-object detection**.

## 👥 Contributors

This project was developed collaboratively by **Group 32**:

* **Smit Patel:**
    * Led data loading, preprocessing, EDA, and bounding box visualization.
    * Implemented MLP and CNN deep learning models using TensorFlow/Keras.
    * Created evaluation logic and visual comparison plots.
* **Shubham Limbachiya:**
    * Focused on model training for traditional ML (KNN, Random Forest, Linear Regression).
    * Managed hyperparameter tuning using GridSearchCV.
    * Contributed to performance evaluation, error analysis, and documentation.

<br>
<details>
<summary><b>Recommended <code>requirements.txt</code> file</b></summary>

```
pandas
seaborn
matplotlib
scikit-learn
tensorflow
gradio
Pillow
```
</details>

<details>
<summary><b>Recommended <code>.gitignore</code> file</b></summary>

```
# Python
__pycache__/
*.pyc
.venv/
venv/
*.env

# Data & Models (These are too large for GitHub)
*.zip
*.joblib
*.keras
*.h5
images_repo/

# OS-specific
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
```
</details>
