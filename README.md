# Perceptron from Statistical Learning

本项目基于李航《统计学习方法》中的感知机（Perceptron）算法，从理论、算法实现到实验验证，完整实现经典感知机模型。

项目主要包含：

- 感知机原始形式
- 感知机对偶形式
- Gram 矩阵实现
- 随机数据实验
- 样本量与参数更新次数实验
- 训练集 / 测试集实验
- 标签噪声实验
- 分类决策边界可视化

---

## 1. 项目背景

感知机是最早的二分类线性分类模型之一，也是理解现代机器学习分类算法的重要基础。

给定训练数据：

$$
D=\{(x_i,y_i)\}_{i=1}^{N}
$$

其中：

$$
x_i\in\mathbb{R}^d,\qquad y_i\in\{-1,+1\}
$$

感知机通过寻找一个超平面：

$$
w^Tx+b=0
$$

将不同类别的样本分开。

对于被错误分类的样本，如果满足：

$$
y_i(w^Tx_i+b)\leq0
$$

则进行参数更新：

$$
w\leftarrow w+y_ix_i
$$

$$
b\leftarrow b+y_i
$$

当所有训练样本都被正确分类时，算法停止。

---

## 2. 项目目标

本项目并不仅仅是实现一个可以运行的 Perceptron，而是希望通过实验理解以下问题：

1. 感知机如何通过误分类样本更新参数？
2. 为什么在线性可分数据上感知机可以收敛？
3. 原始形式与对偶形式有什么区别？
4. Gram 矩阵在对偶形式中起什么作用？
5. 样本数量如何影响参数更新次数？
6. 训练集与测试集之间有什么关系？
7. 当数据存在标签噪声时，感知机的收敛性会发生什么变化？

---

## 3. 算法原理

### 3.1 原始形式

感知机直接维护参数：

$$
w,b
$$

对于第 $i$ 个样本，如果：

$$
y_i(w^Tx_i+b)\leq0
$$

则更新：

$$
w\leftarrow w+y_ix_i
$$

$$
b\leftarrow b+y_i
$$

本项目在 `src/perceptron_original.py` 中实现。

---

### 3.2 对偶形式

感知机还可以表示为：

$$
w=\sum_{i=1}^{N}\alpha_i y_i x_i
$$

其中：

$$
\alpha_i\geq0
$$

表示第 $i$ 个样本参与参数更新的次数。

将其代入分类函数：

$$
w^Tx+b
=
\sum_{j=1}^{N}\alpha_jy_jx_j^Tx+b
$$

因此：

$$
f(x)
=
\sum_{j=1}^{N}\alpha_jy_j(x_j^Tx)+b
$$

为了提高计算效率，可以预先计算 Gram 矩阵：

$$
G_{ij}=x_i^Tx_j
$$

即：

$$
G=XX^T
$$

本项目在 `src/perceptron_dual.py` 中实现感知机对偶形式。

---

## 4. 项目结构

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
│   ├── experiments.py
│   ├── experiment_random.py
│   ├── experiment_sample_size.py
│   ├── experiment_train_test.py
│   └── experiment_noise.py
│
├── notebooks/
│
└── results/
    └── figures/
        ├── perceptron_boundary.png
        ├── random_dataset.png
        ├── updates_vs_sample_size.png
        ├── train_test_accuracy.png
        └── accuracy_vs_noise.png
```

---

# 5. 实验

## 5.1 基础实验：分类决策边界

首先使用简单的二维线性可分数据：

```python
X = np.array([
    [3, 3],
    [4, 3],
    [1, 1],
    [2, 1]
])

y = np.array([1, 1, -1, -1])
```

训练得到：

```text
w = [0. 4.]
b = -6
```

因此决策边界为：

$$
4x_2-6=0
$$

即：

$$
x_2=1.5
$$

模型能够正确分类全部训练样本。

---

## 5.2 随机数据实验

为了验证感知机在随机线性可分数据上的表现，生成两个不同类别的二维随机数据集。

实验记录：

- 参数 $w$
- 偏置 $b$
- Epoch 数
- 参数更新次数
- 分类准确率

实验结果表明，在当前随机数据集上，感知机能够在较少的迭代次数内找到分类边界。

---

## 5.3 样本量实验

通过改变训练样本数量：

```text
20
50
100
200
500
```

观察样本数量对感知机训练过程的影响。

重点考察：

$$
\text{Sample Size}
\rightarrow
\text{Number of Updates}
$$

由于感知机的训练过程依赖误分类样本，因此样本数量、数据分布以及样本顺序都会影响最终的参数更新次数。

---

## 5.4 训练集与测试集实验

进一步将数据划分为训练集和测试集。

训练过程：

$$
X_{train},y_{train}
\rightarrow
\text{Perceptron}
\rightarrow
(w,b)
$$

然后在未参与训练的测试集上进行预测：

$$
\hat y=\operatorname{sign}(w^Tx+b)
$$

分别计算：

$$
Accuracy_{train}
$$

和：

$$
Accuracy_{test}
$$

用于观察模型的泛化表现。

---

## 5.5 标签噪声实验

为了研究数据不可完全线性分离时感知机的表现，在训练集中随机翻转一定比例的标签。

噪声水平设置为：

```text
0%
2%
5%
10%
15%
20%
```

例如：

```python
y_train[noise_indices] *= -1
```

实验结果：

| Noise | Epochs | Updates | Train Accuracy | Test Accuracy |
|---:|---:|---:|---:|---:|
| 0% | 3 | 44 | 1.0000 | 0.9988 |
| 2% | 1000 | 47226 | 0.9740 | 0.9936 |
| 5% | 1000 | 87023 | 0.6900 | 0.6986 |
| 10% | 1000 | 126359 | 0.7760 | 0.8326 |
| 15% | 1000 | 138545 | 0.6260 | 0.6904 |
| 20% | 1000 | 178312 | 0.6640 | 0.7512 |

这里的 `1000 epochs` 表示模型达到人为设置的最大训练轮数，而不是说明算法已经收敛。

### 实验现象

当噪声为 0% 时，训练数据线性可分，感知机能够正常收敛。

当加入标签噪声后，训练数据通常不再严格线性可分，因此感知机可能无法满足：

$$
y_i(w^Tx_i+b)>0
$$

对所有训练样本同时成立。

因此模型会持续发现误分类样本并进行参数更新。

从实验结果可以观察到：

$$
\text{Noise}\uparrow
\quad\Rightarrow\quad
\text{Updates generally increase}
$$

同时，测试准确率开始出现明显波动。

这说明感知机的经典收敛性质依赖于训练数据的线性可分性。

---

# 6. 实验结果可视化

项目将实验生成的图片统一保存到：

```text
results/figures/
```

包括：

- `perceptron_boundary.png`：感知机分类决策边界
- `random_dataset.png`：随机数据集
- `updates_vs_sample_size.png`：样本量与更新次数关系
- `train_test_accuracy.png`：训练集与测试集准确率
- `accuracy_vs_noise.png`：标签噪声与测试准确率

---

# 7. 如何运行

首先进入项目目录：

```bash
cd perceptron-from-statistical-learning
```

安装依赖：

```bash
pip install numpy matplotlib
```

运行基础实验：

```bash
python experiments/experiments.py
```

运行随机数据实验：

```bash
python experiments/experiment_random.py
```

运行样本量实验：

```bash
python experiments/experiment_sample_size.py
```

运行训练集 / 测试集实验：

```bash
python experiments/experiment_train_test.py
```

运行标签噪声实验：

```bash
python experiments/experiment_noise.py
```

实验生成的图片会保存到：

```text
results/figures/
```

---

# 8. 技术栈

- Python
- NumPy
- Matplotlib
- Git
- GitHub

---

# 9. 学习参考

主要参考：

> 李航，《统计学习方法》，人民邮电出版社。

本项目主要对应书中的感知机章节，并在此基础上加入随机数据、训练测试划分以及噪声实验，用于进一步理解算法的训练过程和收敛性质。

---

# 10. 后续可以扩展的方向

后续可以进一步研究：

- 感知机的 Pocket Algorithm
- 非线性分类
- 核方法
- 支持向量机
- Logistic Regression
- 不同优化算法之间的比较
- 更复杂的数据集
- 多分类感知机

---

## License

This project is for learning and research purposes.