# ml_pipeline.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def load_bank_data(csv_path='../../classification_data/bank/bank.csv'):
    """读取并预处理 bank.csv 数据"""
    df = pd.read_csv(csv_path, sep=';')
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    X = df.drop('y', axis=1)
    y = df['y']
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    numerical_cols = X.select_dtypes(include='number').columns.tolist()
    return X, y, categorical_cols, numerical_cols


def build_pipeline(numerical_cols, categorical_cols, model=None):
    """构建预处理 + 模型的 pipeline"""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    if model is None:
        model = DecisionTreeClassifier(max_depth=5, random_state=42)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return pipeline
