# Perceptron From Statistical Learning

本项目基于李航《统计学习方法》中的**感知机（Perceptron）**算法，使用 Python 从零实现感知机的**原始形式（Primal Form）**和**对偶形式（Dual Form）**，并通过二维线性可分数据进行实验与可视化。

项目主要用于理解感知机的数学原理、算法实现，以及原始形式与对偶形式之间的联系。

---

## 1. 项目简介

感知机是一种经典的二分类线性模型，也是机器学习和神经网络领域的重要基础算法。

对于训练数据：

$$
(x_i,y_i),\quad i=1,\dots,n
$$

其中：

$$
x_i\in\mathbb{R}^p,\qquad y_i\in\{-1,+1\}
$$

感知机使用如下线性分类函数：

$$
f(x)=\operatorname{sign}(w^Tx+b)
$$

其中：

- $w$ 为权重向量；
- $b$ 为偏置；
- $y_i$ 为样本的真实标签。

当样本被错误分类时：

$$
y_i(w^Tx_i+b)\leq0
$$

感知机对模型参数进行更新。

---

## 2. 感知机原始形式

### 2.1 模型

感知机的原始形式直接维护权重向量 $w$ 和偏置 $b$。

初始化：

$$
w=0,\qquad b=0
$$

对于被错误分类的样本 $(x_i,y_i)$，执行：

$$
w\leftarrow w+y_ix_i
$$

$$
b\leftarrow b+y_i
$$

不断重复上述过程，直到所有训练样本均被正确分类。

### 2.2 Python 实现

对应代码：

```text
src/perceptron_original.py
```

核心实现：

```python
if y[i] * (np.dot(self.w, X[i]) + self.b) <= 0:

    self.w += y[i] * X[i]
    self.b += y[i]
```

---

## 3. 感知机对偶形式

### 3.1 基本思想

在原始形式中，我们直接维护权重向量 $w$。

对偶形式则利用：

$$
w=\sum_{i=1}^{n}\alpha_i y_i x_i
$$

其中 $\alpha_i$ 表示第 $i$ 个样本参与参数更新的次数。

因此，对偶形式不需要直接更新 $w$，而是维护每个训练样本对应的 $\alpha_i$。

---

### 3.2 分类函数

由于：

$$
w=\sum_{j=1}^{n}\alpha_jy_jx_j
$$

因此：

$$
w^Tx_i
=
\sum_{j=1}^{n}
\alpha_jy_jx_j^Tx_i
$$

于是感知机的分类条件可以写成：

$$
y_i
\left(
\sum_{j=1}^{n}
\alpha_jy_jx_j^Tx_i+b
\right)
\leq0
$$

如果样本被错误分类，则：

$$
\alpha_i\leftarrow\alpha_i+1
$$

同时：

$$
b\leftarrow b+y_i
$$

对应代码：

```text
src/perceptron_dual.py
```

---

## 4. Gram 矩阵

为了计算样本之间的内积，定义 Gram 矩阵：

$$
G_{ij}=x_i^Tx_j
$$

对于训练数据矩阵 $X$：

$$
G=XX^T
$$

本项目中使用 NumPy：

```python
gram = X @ X.T
```

这样就可以利用样本之间的内积计算感知机的分类结果。

---

## 5. 实验数据

本项目使用二维线性可分数据：

$$
X=
\begin{pmatrix}
3&3\\
4&3\\
1&1\\
2&1
\end{pmatrix}
$$

对应标签：

$$
y=
\begin{pmatrix}
1\\
1\\
-1\\
-1
\end{pmatrix}
$$

其中：

- $(3,3)$ 和 $(4,3)$ 为正样本；
- $(1,1)$ 和 $(2,1)$ 为负样本。

因此可以通过一条直线将两类样本分开。

---

## 6. 原始形式实验结果

运行原始形式的感知机后得到：

```text
w = [0. 4.]
b = -6
```

因此：

$$
w=(0,4)
$$

$$
b=-6
$$

感知机的分类函数为：

$$
f(x)=\operatorname{sign}(4x_2-6)
$$

---

## 7. 决策边界

感知机的决策边界满足：

$$
w^Tx+b=0
$$

代入本实验得到的参数：

$$
4x_2-6=0
$$

因此：

$$
\boxed{x_2=1.5}
$$

也就是说，模型学习到了一条水平决策边界，将正样本和负样本分开。

实验生成的决策边界如下：

![Perceptron Decision Boundary](results/figures/perceptron_boundary.png)

---

## 8. 对偶形式实验结果

对偶形式训练后得到：

```text
alpha = [4. 1. 6. 5.]
w = [0. 4.]
b = -6
```

因此：

$$
\alpha=(4,1,6,5)
$$

利用：

$$
w=\sum_{i=1}^{n}\alpha_i y_i x_i
$$

可以恢复得到：

$$
w=(0,4)
$$

与原始形式得到的结果一致。

---

## 9. 两种形式的预测结果

原始形式：

```text
prediction = [ 1.  1. -1. -1.]
true label = [ 1  1 -1 -1]
```

对偶形式：

```text
prediction = [ 1.  1. -1. -1.]
true label = [ 1  1 -1 -1]
```

两种实现均正确分类了全部训练样本。

因此，在本实验数据上：

$$
\boxed{
\text{Original Perceptron}
=
\text{Dual Perceptron}
}
$$

这里的“一致”是指两种实现最终得到相同的分类器，而不是指两种算法的计算过程完全相同。

---

## 10. 项目结构

```text
perceptron-from-statistical-learning/
│
├── README.md
├── .gitignore
│
├── src/
│   ├── perceptron_original.py
│   └── perceptron_dual.py
│
├── experiments/
│   └── experiments.py
│
├── notebooks/
│
└── results/
    └── figures/
        └── perceptron_boundary.png
```

---

## 11. 环境依赖

本项目主要使用：

- Python
- NumPy
- Matplotlib

安装依赖：

```bash
pip install numpy matplotlib
```

---

## 12. 运行实验

在项目根目录运行：

```bash
python experiments/experiments.py
```

程序将：

1. 构造实验数据；
2. 训练原始形式感知机；
3. 训练对偶形式感知机；
4. 输出 $w$、$b$ 和 $\alpha$；
5. 输出预测结果；
6. 绘制决策边界；
7. 将图像保存到：

```text
results/figures/perceptron_boundary.png
```

---

## 13. 学习目标

通过本项目理解以下内容：

1. 感知机的数学模型；
2. 感知机的误分类条件；
3. 感知机参数更新规则；
4. 感知机原始形式的 Python 实现；
5. 感知机对偶形式的数学推导；
6. $\alpha$ 与 $w$ 之间的关系；
7. Gram 矩阵的定义和作用；
8. 线性分类器决策边界的计算；
9. NumPy 向量化计算；
10. 机器学习算法从数学公式到代码实现的过程。

---

## 14. 后续实验

后续计划进一步扩展实验，包括：

- 随机生成更大的二维线性可分数据集；
- 比较不同数据规模下的训练过程；
- 记录感知机迭代次数；
- 比较原始形式与对偶形式的计算过程；
- 增加训练集准确率等评价指标；
- 研究样本顺序对感知机训练结果的影响；
- 尝试核方法，为进一步学习支持向量机（SVM）做准备。

---

## 15. Reference

李航：《统计学习方法》，人民邮电出版社。