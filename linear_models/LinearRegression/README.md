# 📘 Linear Regression 线性回归

## 一、模型简介

线性回归是一种最基本的回归模型， 回归（regression）是能为一个或多个自变量与因变量之间关系建模的一类方法。 在自然科学和社会科学领域，回归经常用来表示输入和输出之间的关系。 
在机器学习领域中的大多数任务通常都与预测（prediction）有关。 当我们想预测一个数值时，就会涉及到回归问题。 
常见的例子包括：预测价格（房屋、股票等）、预测住院时间（针对住院病人等）、 预测需求（零售销量等）。

### ✅ 主要特点
- 易于解释，模型可视化
- 可加正则化（Ridge / Lasso）
- 假设线性关系，抗噪性较差

---

## 二、数学原理（Linear Regression）

---

### 🧩 1. 问题定义

我们有一组训练数据：

$$
\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \dots, (x^{(m)}, y^{(m)})\}
$$

其中：

- $x^{(i)} \in \mathbb{R}^n$：第 $i$ 个样本的输入特征（可多维）  
- $y^{(i)} \in \mathbb{R}$：第 $i$ 个样本的目标值（真实输出）

---

### 📈 2. 模型假设（Hypothesis Function）

我们假设输入与输出之间存在线性关系：

$$
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n = \boldsymbol{\theta}^T \boldsymbol{x}
$$

其中：

- $x = [1, x_1, x_2, \dots, x_n]^T$：加了偏置项的输入向量  
- ${\theta} = [\theta_0, \theta_1, \dots, \theta_n]^T$：模型参数向量

---

### 🎯 3. 损失函数（Loss Function）

我们希望预测值 $\hat{y}$ 尽可能接近真实值 $y$，因此定义均方误差（MSE）作为损失函数：

$$
\hat{y} = \sum_{i=0}^{n} \theta_i x_i = \boldsymbol{\theta}^T \boldsymbol{x}
$$



其中 $h_\theta(x^{(i)})$ 是第 $i$ 个样本的预测值。

> 为什么要除以 2？为了在梯度下降中求导简化公式。

---

### 🧮 4. 求解方法（两种）

#### ✅ 4.1 解析解（Normal Equation）

在特征矩阵 $X$ 满足可逆条件时，可以直接解出参数：

$$
{\theta} = (X^T X)^{-1} X^T y
$$

其中：

- $X \in \mathbb{R}^{m \times (n+1)}$：带有偏置项的特征矩阵  
- $y \in \mathbb{R}^{m \times 1}$：目标值列向量

适合特征维度较小、样本数量不大的情况。

---

#### ✅ 4.2 梯度下降法（Gradient Descent）

使用迭代法反复更新参数来最小化损失函数：

**单个参数的更新公式：**

$$
\theta_j := \theta_j - \eta \cdot \frac{\partial J(\theta)}{\partial \theta_j}
$$

**导数计算：**

$$
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
$$

**更新总结公式：**

$$
\theta_j := \theta_j - \eta \cdot \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
$$

其中：

- $\eta$：学习率，控制每次更新的步长大小

---

### 📐 5. 几何解释

每一组 $(x, y)$ 是一个数据点，线性回归试图找到一个**超平面**，使得所有点到该平面的垂直距离平方和最小（最小二乘意义下的最优拟合）。

> 残差 = 实际值 − 预测值

---

### 📌 6. 为什么叫最小二乘法（OLS）

线性回归也叫普通最小二乘法（Ordinary Least Squares, OLS），它的目标就是最小化所有残差的平方和：

$$
\sum_{i=1}^m \left( y_i - \hat{y}_i \right)^2
$$

这是最常用的误差度量方式，尤其在误差服从高斯分布时非常有效。

---

## 三、代码实战

- 使用 `scikit-learn` 构建模型
- 可视化拟合直线
- 添加噪声 + 对比训练误差 vs 测试误差

👉 示例请查看：[`LinearRegression.ipynb`](./LinearRegression.ipynb)

---

## 四、常见面试问题

1. **为什么线性回归不能用于分类？**
2. **如何解决多重共线性问题？**
3. **线性回归 vs 岭回归 vs Lasso 的区别？**
4. **什么时候使用梯度下降而不是解析解？**

---

## 五、总结与扩展

- 支持多元线性、多项式扩展
- 可以作为更复杂模型（如 GBDT）中的基模型
