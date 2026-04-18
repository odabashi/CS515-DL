# 🧪 MLP Experiments on MNIST

This document summarizes a series of experiments conducted on an MLP model for MNIST classification.  
We analyze the impact of:

- Early stopping
- Network depth
- Neuron width
- Activation functions
- Dropout & BatchNorm
- Optimizers & Regularization

---

# 📌 1. Early Stopping Experiments

## 1.1 Without Early Stopping
- **Architecture:** [256, 128]
- **Epochs:** 10
- **Best Epoch:** 6
- **Train Acc:** ~99.7%
- **Validation Acc:** **97.76%**
- **Train Loss:** ~0.010
- **Validation Loss:** ~0.092
- **Time:** 638s

### Insight
- Model continues training even after convergence → slight overfitting after epoch 6.
- Wasted computation time.

---

## 1.2 With Early Stopping
- **Architecture:** [256, 128]
- **Epochs:** Stopped at 16 (best at 11)
- **Train Acc:** ~100%
- **Validation Acc:** **98.14%**
- **Validation Loss:** ~0.088
- **Time:** 519s

### Insight
- Early stopping improves **generalization and efficiency**.

---

# 📌 2. Depth Experiments

| Layers | Architecture | Best Validation Accuracy | Best Epoch | Time |
|------|-------------|-------------|------------|------|
| 1 | [256] | 97.97% | 8          | 451s |
| 2 | [256,128] | 98.14% | 11         | 519s |
| 3 | [512,256,128] | **98.27%** | 6          | **342s** |
| 4 | [1024,512,256,128] | 97.91% | 6          | 357s |

### Insight
- Increasing depth helps up to **3 layers**.
- 4 layers → **over-parameterization** → worse performance.
- Best trade-off: **3 layers**.

---

# 📌 3. Width (Neuron Count) Experiments

| Architecture | Best Validation Accuracy | Best Epoch | Time |
|-------------|-------------|------------|------|
| [256,128,64] | 97.98% | 6          | 358s |
| [512,256,128] | **98.27%** | 6          | 342s |
| [1024,512,256] | 97.95% | 6          | 412s |

### Insight
- Too small → underfitting.
- Too large → no gain + slower training.
- Sweet spot: **[512,256,128]**.

---

# 📌 4. Activation Function Experiments

| Activation | Best Validation Accuracy | Time |
|-----------|-------------|------|
| ReLU | **98.27%** | 342s |
| Leaky ReLU | 97.82% | 347s |
| Tanh | 98.14% | 569s |
| ELU | 98.12% | **338s** |
| GELU | 98.19% | 443s |

### Insight
- **ReLU remains strongest baseline**.
- GELU competitive but slower.
- Tanh works but is inefficient.
- ELU is a **good fast alternative**.

---

# 📌 5. Dropout Experiments (No BatchNorm)

| Dropout | Best Validation Accuracy | Best Epoch | Time |
|--------|-------------|------------|------|
| 0.0 | 98.27% | 6          | 342s |
| 0.2 | **98.39%** | 11         | 554s |
| 0.5 | 98.23% | 18         | 702s |
| 0.7 | 97.20% | 23         | 855s |

### Insight
- Moderate dropout (0.2) → **best generalization**.
- High dropout → **underfitting + slow training**.

---

# 📌 6. Dropout + BatchNorm

| Setup                  | Best Validation Accuracy | Epoch | Time |
|------------------------|-----------|-------|------|
| Dropout 0.2 without BN | 98.39% | 11    | 554s |
| Dropout 0.2 with BN    | **98.48%** | 12    | 522s |

### Insight
- BatchNorm + Dropout improves stability and performance.
- This is one of the **best-performing configurations**.

---

# 📌 7. Optimizer Comparison

| Optimizer | Best Validation Accuracy | Time |
|----------|-------------|------|
| SGD | 98.06% | 917s |
| Adam | 98.48% | 522s |
| AdamW | **98.48%** | 518s |

### Insight
- Adam/AdamW outperform SGD significantly.
- SGD requires more epochs and time.

---

# 📌 8. Regularization (AdamW)

| L1 | L2 | Best Validation Accuracy | Time |
|----|----|------------|------|
| 0 | 0 | **98.48%** | 518s |
| 1e-3 | 0 | 98.02% | 968s |
| 0 | 1e-3 | **98.45%** | 504s |
| 1e-3 | 1e-3 | 98.12% | 1529s |

### Insight
- **L2 regularization slightly helps**.
- L1 harms performance and slows training.
- Combined L1+L2 → unnecessary complexity.

---

# 🏆 Final Best Configuration

- **Architecture:** [512, 256, 128]
- **Activation:** ReLU
- **Dropout:** 0.2
- **BatchNorm:** Enabled
- **Optimizer:** Adam / AdamW
- **Regularization:** Optional L2 (1e-3)
- **Early Stopping:** Enabled

### Final Performance
- **Validation Accuracy:** **98.48%**
- **Training Time:** 518s = ~8.5 Mins

---

# 🔍 Overall Insights

- Early stopping is **essential** for efficiency.
- Optimal depth exists → deeper ≠ better.
- Moderate regularization (dropout + BN + L2) gives best results.
- Adam/AdamW are **clearly superior** optimizers.
- Simplicity (ReLU + well-sized network) still wins over complex setups.

---

# 📈 Key Takeaway

> The best performance comes from a **balanced model**:
> not too deep, not too wide, and with **controlled regularization**.