# Loan-prediction
Automated Loan Eligibility Prediction system using Machine Learning. This project analyzes applicant data—income, credit history, and demographics—to predict loan approval. Includes a full EDA notebook and a production-ready training script. Built with Scikit-Learn, it features automated preprocessing and multi-model evaluation (Random Forest/SVM).


# Loan Eligibility Prediction System

An end-to-end Machine Learning project to automate the loan approval process based on customer profiles.

## 📌 Project Overview
Loan providers often struggle with manual assessment of loan applications. This project provides a predictive model built with Python to determine loan eligibility. It uses historical customer data to train a classifier that predicts `Loan_Status` (Approved/Rejected) based on various socioeconomic and financial factors.

## 📊 Dataset Features
The model utilizes the following features from the dataset:
- **Loan_ID**: Unique Loan ID
- **Gender**: Male/ Female
- **Married**: Applicant married (Y/N)
- **Dependents**: Number of dependents
- **Education**: Applicant Education (Graduate/ Under Graduate)
- **Self_Employed**: Self-employed (Y/N)
- **ApplicantIncome**: Applicant income
- **CoapplicantIncome**: Coapplicant income
- **LoanAmount**: Loan amount in thousands
- **Loan_Amount_Term**: Term of the loan in months
- **Credit_History**: Credit history meets guidelines (1/0)
- **Property_Area**: Urban/ Semi-Urban/ Rural

## 🛠️ Tech Stack
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Tools:** Jupyter Notebook, Pickle/Joblib (for model serialization)

## 🚀 Key Features
- **Automated Preprocessing:** Handles missing values using median (numerical) and most frequent (categorical) strategies.
- **Feature Engineering:** Implements Label Encoding for categorical data and Standard Scaling for numerical features.
- **Model Comparison:** Trains and evaluates multiple algorithms including **Random Forest**, **Logistic Regression**, and **SVM**.
- **Model Persistence:** Saves the best-performing model (`best_model.pkl`), scalers, and encoders for easy inference.

## 📁 Repository Structure
- `loan_prediction.ipynb`: Complete research and development notebook (EDA, Visualization, and Model Selection).
- `train_model.py`: A production-ready script to retrain the model on any compatible CSV dataset.
- `best_model.pkl`: The serialized high-performance model.
- `scaler.pkl`: Standard scaler used for normalization.
- `label_encoders.pkl` / `encoders.pkl`: Encoders for transforming categorical user input.
- `num_cols.pkl` / `feature_order.pkl`: Metadata to ensure consistent feature engineering during prediction.

## 📈 Performance
The primary model used is a **Random Forest Classifier**, which was selected based on its cross-validation accuracy and F1-score. Evaluation metrics include:
- Accuracy Score
- Precision & Recall
- F1-Score
- 5-Fold Cross-Validation

## ⚙️ How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/Akhil0702/loan-prediction.git](https://github.com/Akhil0702/loan-prediction.git)

   Install dependencies:

2.Bash
pip install -r requirements.txt
Run the training script:

3.Bash
python train_model.py
View the analysis: Open loan_prediction.ipynb in Jupyter Notebook.
