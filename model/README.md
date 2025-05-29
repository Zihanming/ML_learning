# 🧠 ML_learning: Machine Learning Model Collection

This repository provides a curated, modular collection of classical machine learning models implemented in Python. It is structured for both **educational purposes** and **project prototyping**, covering:

- ✅ Supervised learning: classification and regression
- ✅ Unsupervised learning: clustering, dimensionality reduction, anomaly detection
- ✅ Reusable pipeline modules for clean preprocessing


## Project Introduction

    ML_learning/
    ├── model/
    │ ├── supervised_model/
    │ │ ├── classification_model/
    │ │ │ ├── classification_data/
    │ │ │ │ └── bank/
    │ │ │ │ ├── bank.csv
    │ │ │ │ ├── bank-full.csv
    │ │ │ │ ├── bank-names.txt
    │ │ │ ├── Logistic_Regression.ipynb
    │ │ │ ├── Naive_Bayes.ipynb
    │ │ │ ├── SVM.ipynb
    │ │ │ ├── Tree_Classifier.ipynb
    │ │ │ └── data_pipeline.py
    │ │
    │ │ ├── regression_model/
    │ │ │ ├── regression_data/
    │ │ │ │ └── house_price/
    │ │ │ │ ├── train.csv
    │ │ │ │ ├── test.csv
    │ │ │ │ ├── sample_submission.csv
    │ │ │ │ └── data_description.txt
    │ │ │ └── Linear_Regression.ipynb
    │
    │ ├── unsupervised_model/
    │ │ ├── unsup_data/
    │ │ │ └── Country-data.csv
    │ │ ├── Kmeans.ipynb
    │ │ └── unsup_data_pipeline.py
    │ 
    │ ├── deep_learning_model/
    │
    ├── README.md


## 📊 Datasets Used

This repository uses publicly available datasets suitable for both supervised and unsupervised machine learning tasks:

### 🧪 Supervised Learning
- **📂 bank.csv**  
  From the UCI Bank Marketing dataset.  
  Task: Binary classification (`y`: subscribed to term deposit).  
  Location: `classification_model/classification_data/bank/`

- **📂 house_price**  
  From Kaggle’s House Prices: Advanced Regression Techniques.  
  Task: Regression (predicting house sale prices).  
  Location: `regression_model/regression_data/house_price/`

### 🔍 Unsupervised Learning
- **📂 Country-data.csv**  
  From Kaggle: *Unsupervised Learning on Country Data*  
  Task: Clustering countries based on socio-economic indicators.  
  Location: `unsupervised_model/unsup_data/`

All datasets are preprocessed using custom pipeline scripts (`data_pipeline.py` and `unsup_data_pipeline.py`) and split into train/test if necessary.


## 🔧 Pipeline Modules

To ensure consistent preprocessing across models, the project includes reusable pipeline modules built with `scikit-learn`’s `Pipeline` and `ColumnTransformer`.

For example

### 📦 `data_pipeline.py`
Used in: **classification & regression tasks**  
Located at: `classification_model/data_pipeline.py`

**Key Features:**
- Automatically encodes categorical columns with `OneHotEncoder`
- Scales numerical columns with `StandardScaler`
- Easily wraps any estimator (e.g., LogisticRegression, DecisionTree)

They works in similar for different model

