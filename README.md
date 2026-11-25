# 🎓 Student Exam Performance Indicator — End-to-End Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![AWS Elastic Beanstalk](https://img.shields.io/badge/AWS-Elastic%20Beanstalk-FF9900?logo=amazonwebservices&logoColor=white)

An end-to-end Machine Learning web application that predicts a student's **Mathematics exam score (`math_score`)** based on demographic background, parental education, lunch type, test preparation course, and reading/writing scores.

Built with a production-grade modular pipeline architecture (`Data Ingestion` → `Data Transformation` → `Model Trainer` → `Prediction Pipeline`), a modern **Flask** web interface, timestamped logging, custom exception handling, and ready-to-deploy configuration for **AWS Elastic Beanstalk**.

---

## 📌 Problem Statement

Understanding how student exam performance is influenced by factors such as gender, ethnicity, parental level of education, lunch status, and test preparation course—and accurately predicting a student's **Math Score** in real time through a web application.

---

## 📊 Dataset Overview

- **Source:** [Students Performance in Exams (Kaggle)](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Size:** 1,000 rows × 8 columns (`notebook/data/stud.csv`)

| Feature | Type | Description |
| :--- | :--- | :--- |
| `gender` | Categorical | Student's gender (`male`, `female`) |
| `race_ethnicity` | Categorical | Ethnic group (`group A` – `group E`) |
| `parental_level_of_education` | Categorical | Highest education level of parents (`some high school` to `master's degree`) |
| `lunch` | Categorical | Lunch type (`standard`, `free/reduced`) |
| `test_preparation_course` | Categorical | Completion status of test prep course (`none`, `completed`) |
| `reading_score` | Numerical | Reading exam score (`0` – `100`) |
| `writing_score` | Numerical | Writing exam score (`0` – `100`) |
| **`math_score`** *(Target)* | **Numerical** | **Target variable to predict (`0` – `100`)** |

---

## 🧠 Machine Learning Pipeline & Model Performance

### Preprocessing (`src/components/data_transformation.py`)
- **Numerical Pipeline:** `SimpleImputer(strategy="median")` → `StandardScaler()`
- **Categorical Pipeline:** `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder()` → `StandardScaler(with_mean=False)`
- Combined via `ColumnTransformer` and serialized into `artifacts/proprocessor.pkl`.

### Model Training & Evaluation (`src/components/model_trainer.py`)
Multiple regression models were trained and tuned using `GridSearchCV` (3-fold CV), automatically selecting and saving the highest-scoring model to `artifacts/model.pkl`:

| Model | Test $R^2$ Score |
| :--- | :---: |
| **Linear Regression / Ridge** | **~0.8804 – 0.8806** *(Best)* |
| CatBoost Regressor | ~0.8516 |
| Random Forest Regressor | ~0.8485 |
| AdaBoost Regressor | ~0.8473 |
| XGBoost Regressor | ~0.8278 |
| Decision Tree Regressor | ~0.7317 |

---

## 🗂️ Project Structure

```text
mlproject-by-Gio/
├── .ebextensions/
│   └── python.config                  # AWS Elastic Beanstalk WSGI configuration
├── artifacts/
│   ├── data.csv                       # Raw dataset snapshot
│   ├── train.csv                      # Training split (80%)
│   ├── test.csv                       # Testing split (20%)
│   ├── proprocessor.pkl               # Fitted preprocessing pipeline
│   └── model.pkl                      # Best trained ML model
├── notebook/
│   ├── data/stud.csv                  # Original dataset
│   ├── 1 . EDA STUDENT PERFORMANCE .ipynb
│   └── 2. MODEL TRAINING.ipynb
├── src/
│   ├── components/
│   │   ├── data_ingestion.py          # Data loading & train/test split
│   │   ├── data_transformation.py     # Feature engineering & preprocessing
│   │   └── model_trainer.py           # Model training, tuning & export
│   ├── pipeline/
│   │   ├── train_pipeline.py          # Training orchestration
│   │   └── predict_pipeline.py        # Inference pipeline & CustomData schema
│   ├── exception.py                   # Custom exception with script/line tracking
│   ├── logger.py                      # Timestamped folder-per-run logging
│   └── utils.py                       # Serialization (dill) & model evaluation helpers
├── templates/
│   ├── index.html                     # Modern landing page
│   └── home.html                      # Student performance prediction form & results UI
├── app.py                             # Local development Flask entry point
├── application.py                     # Production Flask entry point (AWS Elastic Beanstalk)
├── requirements.txt                   # Project dependencies
├── setup.py                           # Local package setup
└── README.md
```

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data & ML:** Pandas, NumPy, Scikit-Learn, XGBoost, CatBoost, Dill
- **Visualization (EDA):** Matplotlib, Seaborn
- **Web Framework & UI:** Flask, HTML5, CSS3 (Responsive Dark Glassmorphism UI)
- **Cloud Deployment:** AWS Elastic Beanstalk & AWS CodePipeline (CI/CD via GitHub)

---

## 🚀 Getting Started (Run Locally)

### 1. Clone the Repository
```bash
git clone https://github.com/Gio365/mlproject-by-Gio.git
cd mlproject-by-Gio
```

### 2. Create and Activate Environment
Using **Conda**:
```bash
conda create -p venv python=3.13 -y
conda activate ./venv
```

*(Or using standard Python `venv`)*:
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS / Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Training Pipeline *(Optional — artifacts already included)*
```bash
python src/components/data_ingestion.py
```
This runs Data Ingestion → Data Transformation → Model Trainer and generates `artifacts/model.pkl` and `artifacts/proprocessor.pkl` along with execution logs under `logs/`.

### 5. Launch the Flask Web App
```bash
python app.py
```
Open your browser and visit:
- **Landing Page:** `http://127.0.0.1:5000/`
- **Prediction Form:** `http://127.0.0.1:5000/predictdata`

---

## ☁️ Deployment on AWS Elastic Beanstalk

This repository is pre-configured for **AWS Elastic Beanstalk** deployment:
- **`application.py`** serves as the production WSGI entry point (`application = Flask(__name__)`).
- **`.ebextensions/python.config`** configures `WSGIPath: application:application`.
- **`artifacts/`** contains the pre-trained `model.pkl` and `proprocessor.pkl` so inference works out of the box after deployment.

### Quick Deploy Steps (GitHub + AWS CodePipeline):
1. Create a **Python Web Server Environment** in **AWS Elastic Beanstalk**.
2. Create a pipeline in **AWS CodePipeline** connected to this GitHub repository (`main` branch).
3. Skip the Build stage and set the **Deploy** stage to your Elastic Beanstalk application & environment.
4. Once deployed, access your live app via the Elastic Beanstalk domain URL!

---

## 👤 Author

**Gio**
- GitHub: [@Gio365](https://github.com/Gio365)
- Email: `giovanni.nadika@gmail.com`
