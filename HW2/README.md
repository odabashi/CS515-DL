# 📌 Part 1: Transfer Learning on CIFAR-10 (VGG-16)

This section presents the results of two transfer learning strategies using a pretrained VGG-16 model on CIFAR-10:

- **Option 1:** Frozen feature extractor + input resizing
- **Option 2:** Full fine-tuning without resizing (adaptive architecture)

---

## 🧪 Experiment 1 — Frozen Features + Resized Input (224×224)

### ⚙️ Configuration

- Pretrained: ✅ (ImageNet)
- Freeze Features: ✅
- Input Resize: 32×32 → 224×224
- Trainable Layers: Classifier only
- Optimizer: AdamW
- Learning Rate: 0.001
- Batch Size: 64
- Epochs: 30 (Early stopping at epoch 20)
- Patience: 5

---

### 📈 Training Summary

| Metric | Value                  |
|------|------------------------|
| Best Validation Accuracy | **34.54%** (@Epoch 15) |
| Final Training Accuracy | 26.79%                 |
| Final Training Loss | 2.0141                 |
| Validation Loss (best) | 2.0886                 |
| Training Time | **3013.25s (~50 min)** |

---

### 🧪 Test Performance

| Metric | Value |
|------|------|
| Accuracy | **34.13%** |
| Precision | 0.4184 |
| Recall | 0.3413 |
| F1 Score | 0.2632 |

---

### 📊 Observations

- Model shows **clear underfitting**
- Freezing convolutional layers limits adaptation to CIFAR-10
- Resizing introduces **no new information**, only interpolation
- Strong **prediction bias toward few classes** (7, 8, 9) (See Confusion Matrix.)
- Many classes are nearly ignored (e.g., class 5 = 0%)

---

### ⚠️ Key Insight

> Using pretrained features as-is is insufficient when the domain (low-resolution CIFAR images) differs significantly from ImageNet.

---

## 🧪 Experiment 2 — Full Fine-Tuning (No Resizing (Original Size of 32x32), Adaptive Pooling)

### ⚙️ Configuration

- Pretrained: ✅ (ImageNet)
- Freeze Features: ❌ (All layers trainable)
- Input Resize: ❌ (kept at 32×32)
- Architecture Change:
  - `AdaptiveAvgPool2d(1,1)`
  - Classifier reduced: `512 → 512 → 10`
- Optimizer: AdamW
- Learning Rate: 0.001
- Batch Size: 64
- Epochs: 30
- Patience: 5

---

### 📈 Training Summary

| Metric                      | Value |
|-----------------------------|------|
| Best Training Accuracy     | **96.99%** |
| Best Training Loss         | 0.0910 |
| Best Validation Accuracy    | **95.78%** |
| Best Validation Loss (best) | 0.1480 |
| Training Time               | **1553.60s (~26 min)** |

---

### 🧪 Test Performance

| Metric | Value |
|------|------|
| Accuracy | **87.96%** |
| Precision | 0.8797 |
| Recall | 0.8796 |
| F1 Score | 0.8796 |

---

### 📊 Observations

- Model achieves **strong generalization performance**
- Training converges **quickly and smoothly**
- All classes are **well recognized**, no severe bias
- Confusion matrix shows **balanced predictions**
- Training is **~2x faster** than Option 1

---

### ⚠️ Key Insight

> Allowing the network to adapt its feature representations is critical when transferring to a new dataset with different characteristics.

---

## 🔬 Comparison of Both Experiments

| Metric | Option 1 (Freezing + <br>Resizing to 224x224) | Option 2 (Full Fine-Tuning + <br>Original Size (32x32)) |
|------|-----------------------------------------------|-----------------------------------------------------|
| Test Accuracy | 34.13%                                        | **87.96%**                                          |
| Precision | 0.4184                                        | **0.8797**                                          |
| Recall | 0.3413                                        | **0.8796**                                          |
| F1 Score | 0.2632                                        | **0.8796**                                          |
| Training Time | ~3013s                                        | **~1554s**                                          |
| Convergence | Slow, unstable                                | Fast, stable                                        |
| Bias | High                                          | Low                                                 |

---

## 🧠 Final Insights

- **Freezing layers (Option 1)**:
  - Computationally simple
  - Fails to adapt to CIFAR-10
  - Leads to underfitting and biased predictions

- **Full fine-tuning (Option 2)**:
  - Significantly better performance
  - Learns dataset-specific features
  - More efficient despite training more parameters

---

## 📊 t-SNE Visualization Insight

- **Option 1:** Highly overlapping clusters → poor feature separability  
- **Option 2:** Clear cluster boundaries → strong learned representations  

> This confirms that fine-tuning enables the model to learn a more discriminative feature space.

---

## ✅ Conclusion

Fine-tuning the entire pretrained model while adapting its architecture to small input sizes leads to **dramatic improvements** in performance, stability, and feature representation quality.

Option 2 is clearly the **preferred transfer learning strategy** for CIFAR-10.

---

# 📊 Part 2: Knowledge Distillation Experiments (CIFAR-10)

This section presents a comprehensive comparison of different training strategies:

* Baseline CNN
* ResNet (with and without Label Smoothing)
* Knowledge Distillation (KD)
* Custom KD with MobileNet

---

## 🔹 Experiment 1: Simple CNN (Baseline)

**Configuration**

* Model: CNN
* Label Smoothing: ❌
* Knowledge Distillation: ❌

**Model Complexity**

* FLOPs: **6.28 MMac**
* Parameters: **545K**

**Performance**

| Metric         | Value      |
| -------------- | ---------- |
| Train Accuracy | ~75.7%     |
| Train Loss     | ~0.70      |
| Validation Accuracy   | **78.47%** |
| Validation Loss       | **0.6169** |
| Test Accuracy  | **75.60%** |

**Training**

* Epochs: 30
* Time: ~946s

---

## 🔹 Experiment 2: ResNet (Baseline)

**Configuration**

* Model: ResNet
* Label Smoothing: ❌
* Knowledge Distillation: ❌

**Model Complexity**

* FLOPs: **557.22 MMac**
* Parameters: **11.17M**

**Performance**

| Metric         | Value      |
| -------------- | ---------- |
| Train Accuracy | ~97.75%    |
| Train Loss     | ~0.065     |
| Validation Accuracy   | **97.48%** |
| Validation Loss       | **0.0861** |
| Test Accuracy  | **90.58%** |

**Training**

* Epochs: 30
* Time: ~2562s

---

## 🔹 Experiment 3: ResNet + Label Smoothing (0.1)

**Configuration**

* Model: ResNet
* Label Smoothing: ✅ (0.1)
* Knowledge Distillation: ❌

**Model Complexity**

* FLOPs: **557.22 MMac**
* Parameters: **11.17M**

**Performance**

| Metric         | Value      |
| -------------- | ---------- |
| Train Accuracy | ~98.27%    |
| Train Loss     | ~0.55      |
| Validation Accuracy   | **97.81%** |
| Validation Loss       | **0.5642** |
| Test Accuracy  | **91.38%** |

**Training**

* Epochs: 30
* Time: ~2331s

---

## 🔹 Experiment 4: CNN + Knowledge Distillation (ResNet Teacher)

**Configuration**

* Model: CNN (Student)
* Teacher: ResNet (no LS)
* Label Smoothing: ❌
* Knowledge Distillation: ✅ (standard)

**Model Complexity**

* FLOPs: **6.28 MMac**
* Parameters: **545K**

**Performance**

| Metric         | Value      |
| -------------- | ---------- |
| Train Accuracy | ~73.9%     |
| Train Loss     | ~0.78      |
| Validation Accuracy   | **76.33%** |
| Validation Loss       | **0.8031** |
| Test Accuracy  | **74.81%** |

**Training**

* Epochs: 30
* Time: ~1194s

---

## 🔹 Experiment 5: MobileNet + Custom KD (ResNet + LS Teacher)

**Configuration**

* Model: MobileNet (Student)
* Teacher: ResNet + Label Smoothing (0.1)
* Knowledge Distillation: ✅ (custom)
* KD Mode: True-class probability shaping

**Model Complexity**

* FLOPs: **96.16 MMac**
* Parameters: **2.3M**

**Performance**

| Metric         | Value      |
| -------------- | ---------- |
| Train Accuracy | ~94.6%     |
| Train Loss     | ~0.19      |
| Validation Accuracy   | **94.96%** |
| Validation Loss       | **0.2255** |
| Test Accuracy  | **89.55%** |

**Training**

* Epochs: 30
* Time: ~4306s

---

# 📈 Overall Comparison

| Model                   | FLOPs | Params | Best Validation Accuracy | Test Accuracy  |
| ----------------------- | ----- | ------ |--------------------------| --------- |
| CNN                     | 6.28M | 0.55M  | 78.47%                   | 75.6%     |
| ResNet                  | 557M  | 11.17M | 97.48%                   | 90.58%    |
| ResNet + LS             | 557M  | 11.17M | **97.81%**               | **91.38%** |
| CNN + KD                | 6.28M | 0.55M  | 76.33%                   | 74.81%    |
| MobileNet + KD (Custom) | 96M   | 2.3M   | 94.96%                   | 89.55%    |

---

# 🔍 Key Insights

### 1. Label Smoothing Improves Generalization

* ResNet + LS outperforms vanilla ResNet (+0.8%)
* Slightly higher validation loss but better test accuracy → **better calibration**

---

### 2. Knowledge Distillation is Not Always Beneficial (for weak students)

* CNN + KD performed **slightly worse** than baseline CNN
* Indicates:

  * Student capacity too low
  * Teacher knowledge not effectively utilized

---

### 3. Architecture Matters More Than KD Alone

* CNN (with or without KD) is far behind ResNet
* Capacity gap dominates performance

---

### 4. Custom KD + Better Student (MobileNet) Works Well

* MobileNet + KD achieves **89.55%**
* Very close to ResNet (90.58%) with:

  * **~6x fewer FLOPs**
  * **~5x fewer parameters**

👉 This is the most **efficient trade-off**

---

### 5. Best Trade-offs

| Goal             | Best Choice           |
| ---------------- | --------------------- |
| Highest Accuracy | ResNet + LS           |
| Efficiency       | MobileNet + Custom KD |
| Simplicity       | CNN baseline          |

---

# 🧠 Final Takeaway

* **Label Smoothing** improves robustness and generalization.
* **Knowledge Distillation** is effective **only when the student has enough capacity**.
* **Custom KD + MobileNet** provides the best balance between performance and efficiency.
* Large models (ResNet) still dominate in raw accuracy, but **distillation narrows the gap significantly**.
