# 🧠 Machine Learning Basics

本项目包含机器学习中最核心的基础知识与模块实现，适用于秋招复习、自主学习或面试准备。内容按照模块功能划分为五个子目录，每个子目录中包含 `.ipynb` 笔记、手撕实现与实战示例。

---

## 📁 项目结构说明

```text
basic/
├── data_preprocessing/          # 数据预处理相关
├── generalization_and_tuning/   # 泛化能力与调参技巧
├── model_building_blocks/       # 模型构建基础组件（损失函数、评估指标等）
├── model_core/                  # 常见核心模型（线性、逻辑回归等）
├── pipeline_and_application/    # 实践中的模型部署、管道构建与可解释性
```

## 📦 模块内容概览

---


---

## 🧩 I. 模型核心基础（model_building_blocks）

| 主题                  | 子主题                                                                 |
|---------------------|--------------------------------------------------------------------------|
| Activation Functions | Sigmoid, Tanh, ReLU, LeakyReLU, Softmax, GELU                         |
| Distance Metrics    | Euclidean, Manhattan, Cosine, Mahalanobis                              |
| Loss Functions       | MSE, MAE, Cross-Entropy, Hinge, Log Loss                                 |
| Evaluation Metrics   | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, RMSE         |
| Optimization         | Gradient Descent, SGD, Mini-batch, Momentum, Adam, Learning Rate Schedulers |
| Regularization       | L1 (Lasso), L2 (Ridge), Dropout, Early Stopping                          |

---

## 🧪 II. 数据预处理（data_preprocessing）

| 主题                                   | 子主题                                                           |
|--------------------------------------|------------------------------------------------------------------|
| Missing Value                        | Drop, Imputation (Mean, Median, KNN)                             |
| Scaling & Transformation             | Standardization, Min-Max Scaling, One-Hot Encoding, Binning       |
| Selection & Dimensionality Reduction | Correlation Analysis, PCA, Mutual Information                    |
| Pipeline Construction                | Scikit-learn Pipelines, Transformers                            |
---

## 🧬 III. 泛化与调优（generalization_and_tuning）

| 主题                        | 子主题                                                      |
|---------------------------|-------------------------------------------------------------|
| Overfitting / Underfitting | Bias-Variance Tradeoff, Model Complexity                    |
| Cross Validation           | K-Fold, Stratified, Leave-One-Out                           |
| Hyperparameter Tuning      | Grid Search, Random Search, Bayesian Optimization           |

---

## 📦 IV. 模型上线（application）

| 主题                   | 子主题                                                           |
|----------------------|------------------------------------------------------------------|
| Deployment Concepts   | Inference, Latency, Model Versioning, Drift Detection           |
| Interpretability      | SHAP, LIME, Feature Importance                                  |

---
