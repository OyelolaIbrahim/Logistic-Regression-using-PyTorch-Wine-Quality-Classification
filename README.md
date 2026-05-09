# Logistic Regression using PyTorch — Wine Quality Classification

Implementing logistic regression from scratch in PyTorch 
to classify red wine quality into 6 categories, 
demonstrating core PyTorch concepts: custom Dataset class, 
DataLoader, nn.Module model, CrossEntropyLoss, and 
SGD optimiser — all without using Scikit-Learn's 
LogisticRegression.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![Task](https://img.shields.io/badge/Task-Multi--Class%20Classification-green)
![Dataset](https://img.shields.io/badge/Dataset-Wine%20Quality-red)

---

## Overview

Builds a logistic regression classifier using raw PyTorch 
primitives on the Red Wine Quality dataset. Rather than 
using Scikit-Learn's `LogisticRegression`, every component 
is implemented manually — a custom `Dataset` class, 
`DataLoader` with batching and shuffling, a two-layer 
`nn.Sequential` model, `CrossEntropyLoss`, and `SGD` 
optimiser — demonstrating a deep understanding of 
PyTorch's low-level training loop.

---

## Dataset

- **Name:** Red Wine Quality Dataset
- **File:** `winequality-red.csv`
- **Source:** [Download from Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)
- **Records:** 1,599 red wine samples
- **Input Features:** 11 physicochemical measurements
  (fixed acidity, volatile acidity, citric acid, 
  residual sugar, chlorides, free sulfur dioxide, 
  total sulfur dioxide, density, pH, sulphates, alcohol)
- **Target:** `quality` — wine quality score (3–8), 
  re-indexed to (0–5) by subtracting 3:
```python
  data["quality"] = data["quality"] - 3
```
- **Classes:** 6 quality levels (0 = lowest, 5 = highest)
- **Instructions:** Download `winequality-red.csv` 
  from the link above and place it in the root 
  folder before running the notebook

---

## Approach

### Configuration (Config Class)
All hyperparameters centralised in one place:

| Parameter | Value |
|-----------|-------|
| Batch Size | 10 |
| Input Shape | 11 features |
| Hidden Shape | 20 units |
| Output Shape | 6 classes |
| Learning Rate | 1e-5 |
| Epochs | 5 |

### Custom Dataset Class
Built a `Dataset` class implementing the PyTorch 
dataset interface:
- Converts feature columns to `torch.float32` tensor
- Converts quality labels to `torch.long` tensor 
  (required by `CrossEntropyLoss`)
- Returns `{'x': features, 'y': label}` 
  per sample via `__getitem__`

### Model Architecture
```python
class Logistic_regression(nn.Module):
    self.Layers = nn.Sequential(
        nn.Linear(11, 20),   # input → hidden
        nn.Linear(20, 6)     # hidden → 6 class output
    )
```

| Layer | Input | Output |
|-------|-------|--------|
| Linear 1 | 11 features | 20 hidden units |
| Linear 2 | 20 hidden units | 6 class logits |

### Training Loop
```python
for epoch in range(5):
    for batch in train_loader:
        output = lr_model(batch['x'])
        loss = criterion(output, batch['y'])
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
```


### Evaluation
Custom `metrics()` function computes per-epoch:
- Accuracy, Precision, Recall, F1-Score 
  (weighted average, via Scikit-Learn)
- Predictions obtained using 
  `torch.argmax(torch.softmax(output))`

---

## Results

| Metric | Value |
|--------|-------|
| Epochs | 5 |
| Final Avg Loss | ~9.36 (per epoch) |
| Accuracy | 0.11 |
| Precision | 1.0 |
| Recall | 0.11 |
| F1-Score | 0.20 |
.

---

## Technologies Used

Python, PyTorch (nn.Module, DataLoader, 
CrossEntropyLoss, SGD), Pandas, 
Scikit-Learn (metrics)

---

## How to Run

```bash
git clone https://github.com/OyelolaIbrahim/logistic-regression-pytorch.git
cd logistic-regression-pytorch
pip install -r requirements.txt
jupyter notebook logistic_regression_torch.ipynb
```

---

## Key Takeaways

- PyTorch's `nn.CrossEntropyLoss` internally 
  applies Softmax — do not add a Softmax layer 
  before passing logits to the loss function
- `torch.long` dtype is required for class 
  labels when using `CrossEntropyLoss` — 
  using `float32` will throw a runtime error
- `optimizer.zero_grad()` must be called 
  **before** `loss.backward()` to reset 
  gradients from the previous batch
- With only 5 epochs and no feature scaling, 
  the model underfits — adding `MinMaxScaler` 
  and training for more epochs would 
  significantly improve results

