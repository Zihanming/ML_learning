import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def load_country_data(csv_path="../../Country-data.csv"):
    df = pd.read_csv(csv_path)
    features = df.drop(columns=['country'])
    return df, features

def build_unsupervised_pipeline():
    """只包含数值标准化的 ColumnTransformer（可以用于聚类、降维等）"""
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, slice(0, 9))  # 9个特征
    ])
    return preprocessor
