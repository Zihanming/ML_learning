# ml_pipeline.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt


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

def evaluate_model(model_pipeline, X_test, y_test, model_name="Model"):
    """
    通用分类模型评估函数：打印报告 + 混淆矩阵 + AUC 曲线
    """
    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]

    # 打印评估指标
    print(f" Classification Report ({model_name}):\n", classification_report(y_test, y_pred))
    print(" Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f" Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # 绘制 ROC 曲线
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"{model_name} AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()