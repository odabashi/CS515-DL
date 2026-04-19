# 🧪 Robustness Experiments on CIFAR-10

In this assignment we present a comprehensive evaluation of model robustness under:
- Natural distribution shifts (CIFAR-10-C)
- Adversarial perturbations (PGD)
- Robust training (AugMix)
- Knowledge distillation


# 📌 Experiment 1 — Baseline Model  
**Model:** ResNet (trained from scratch)  
**Setting:** Label Smoothing = 0.1  
**Augmentation:** Standard (no AugMix)


## 🔹 Clean Performance
- Test Accuracy: **0.9138**
- Precision: 0.9138  
- Recall: 0.9138  
- F1 Score: 0.9136  


## 🔹 CIFAR-10-C Corruption Robustness

| Corruption | Accuracy |
|------------|---------|
| gaussian_noise | 0.4655 |
| shot_noise | 0.5831 |
| impulse_noise | 0.5573 |
| speckle_noise | 0.6186 |
| defocus_blur | 0.7757 |
| glass_blur | 0.4568 |
| motion_blur | 0.6995 |
| zoom_blur | 0.7099 |
| gaussian_blur | 0.6889 |
| snow | 0.7441 |
| frost | 0.7239 |
| fog | 0.8177 |
| brightness | 0.8886 |
| contrast | 0.6977 |
| elastic_transform | 0.7801 |
| pixelate | 0.6937 |
| jpeg_compression | 0.7702 |
| spatter | 0.7975 |
| saturate | 0.8568 |

- **Mean Corruption Accuracy (mCA):** 0.7013  
- **Mean Corruption Error    (mCE):** 0.2987  

### 🔍 Detailed Observations
- **Noise corruptions are the weakest point**
  - Gaussian Noise: 0.4655
  - Glass Blur: 0.4568
- **Global transformations are handled better**
  - Brightness: 0.8886
  - Fog: 0.8177
- **Mid-level robustness for blur and compression**

👉 **Conclusion:** Model relies heavily on fine-grained pixel structure → sensitive to noise.


## 🔹 Adversarial Robustness (PGD-20)

| Norm | ε | Clean Accuracy | Adversarial Accuracy | Drop |
|------|--|----------|--------|------|
| L2 | 0.25 | 0.9138 | 0.3357 | -0.5781 |
| L∞ | 4/255 | 0.9138 | 0.1743 | -0.7395 |

👉 L∞ is significantly stronger → many small pixel changes are more destructive than global perturbation.

## 🔹 Visual Analysis

- t-SNE:
  - [L2](./results/exp1/tsne_adversarial_l2.png)
  - [L∞](./results/exp1/tsne_adversarial_l-inf.png)

- Grad-CAM:
  - [L2](./results/exp1/gradcam_l2.png)
  - [L∞](./results/exp1/gradcam_l-inf.png)

👉 **Insights:**
- t-SNE → adversarial samples move into incorrect clusters  
- Grad-CAM → attention shifts from object to irrelevant regions  

---

# 📌 Experiment 2 — AugMix Model  
**Model:** ResNet  
**Setting:** Label Smoothing = 0.1 + AugMix (JSD, λ=12)  
**FLOPs:** 557.22 MMac  
**Parameters:** 11.17M  


## 🔹 Clean Performance
- Test Accuracy: **0.9228**

👉 Slight improvement over baseline.

## 🔹 CIFAR-10-C Corruption Robustness

| Corruption | Accuracy | Δ vs Baseline |
|------------|---------|--------------|
| gaussian_noise | 0.7145 | +0.2490 |
| shot_noise | 0.7898 | +0.2067 |
| impulse_noise | 0.7677 | +0.2104 |
| speckle_noise | 0.8125 | +0.1939 |
| defocus_blur | 0.8927 | +0.1170 |
| glass_blur | 0.6527 | +0.1959 |
| motion_blur | 0.8550 | +0.1555 |
| zoom_blur | 0.8725 | +0.1626 |
| gaussian_blur | 0.8746 | +0.1857 |
| snow | 0.8320 | +0.0879 |
| frost | 0.8257 | +0.1018 |
| fog | 0.8675 | +0.0498 |
| brightness | 0.9094 | +0.0208 |
| contrast | 0.8104 | +0.1127 |
| elastic_transform | 0.8573 | +0.0772 |
| pixelate | 0.8322 | +0.1385 |
| jpeg_compression | 0.8395 | +0.0693 |
| spatter | 0.8755 | +0.0780 |
| saturate | 0.8924 | +0.0356 |

- **Mean Corruption Accuracy (mCA):** 0.8302 (**+0.1289**)  
- **Mean Corruption Error    (mCE):** 0.1698  


### 🔍 Detailed Observations
- **Huge gains in noise robustness (~+20–25%)**
- Improvements across *all corruption types*
- Smaller gains for already strong categories (brightness, fog)

👉 AugMix enforces invariance → significantly improves generalization under distribution shift.


## 🔹 Adversarial Robustness (PGD-20)

| Norm | Clean Accuracy | Adversarial Accuracy | Drop |
|------|----------|--------|------|
| L2 | 0.9228 | 0.2279 | -0.6949 |
| L∞ | 0.9228 | 0.0251 | -0.8977 |


### 🔍 Critical Observation

Compared to baseline:

| Model | L2 Adversarial Accuracy | L∞ Adversarial Accuracy |
|------|-------------------------|--------|
| Baseline | 0.3357                  | 0.1743 |
| AugMix | 0.2279                  | 0.0251 |

👉 **Conclusion:**
- AugMix **reduces adversarial robustness**
- Strong trade-off:
  - ✔ Better corruption robustness  
  - ✖ Worse adversarial robustness  


## 🔹 Visual Analysis

- t-SNE:
  - [L2](./results/exp2/tsne_adversarial_l2.png)
  - [L∞](./results/exp2/tsne_adversarial_l-inf.png)

- Grad-CAM:
  - [L2](./results/exp2/gradcam_l2.png)
  - [L∞](./results/exp2/gradcam_l-inf.png)


---

# 📌 Experiment 3 — Knowledge Distillation  
**Teacher:** ResNet + AugMix  
**Student:** MobileNetV2  


## 🔹 Clean Performance
- Test Accuracy: **0.8900**


## 🔹 CIFAR-10-C Corruption Robustness

| Corruption | Accuracy |
|------------|---------|
| gaussian_noise | 0.4263 |
| shot_noise | 0.5378 |
| impulse_noise | 0.5362 |
| speckle_noise | 0.5725 |
| defocus_blur | 0.7613 |
| glass_blur | 0.4478 |
| motion_blur | 0.6997 |
| zoom_blur | 0.7001 |
| gaussian_blur | 0.6820 |
| snow | 0.7302 |
| frost | 0.6979 |
| fog | 0.8009 |
| brightness | 0.8733 |
| contrast | 0.6457 |
| elastic_transform | 0.7728 |
| pixelate | 0.7092 |
| jpeg_compression | 0.7774 |
| spatter | 0.7897 |
| saturate | 0.8443 |

- **Mean Corruption Accuracy (mCA):** 0.6845  
- **Mean Corruption Error    (mCE):** 0.3155  


### 🔍 Observation
- Performance is **closer to baseline than AugMix teacher**
- Robustness **not transferred effectively**


## 🔹 Adversarial Robustness

| Norm | Clean Accuracy | Adversarial Accuracy | Drop |
|------|----------|--------|------|
| L2 | 0.8900 | 0.2430 | -0.6470 |
| L∞ | 0.8900 | 0.0742 | -0.8158 |


### 🔍 Comparison

| Model | L∞ Adversarial Accuracy |
|------|-----------|
| Baseline | 0.1743 |
| AugMix | 0.0251 |
| Student | 0.0742 |

👉 **Conclusion:**
- Distillation slightly improves over AugMix under attack
- But still far from robust


## 🔹 Visual Analysis

- t-SNE:
  - [L2](./results/exp3/tsne_adversarial_l2.png)
  - [L∞](./results/exp3/tsne_adversarial_l-inf.png)

- Grad-CAM:
  - [L2](./results/exp3/gradcam_l2.png)
  - [L∞](./results/exp3/gradcam_l-inf.png)

---

# 📊 Final Summary

| Model | Clean Accuracy | Mean Corruption Accuracy (mCA) | L∞ Adversarial |
|------|----------|------|--------|
| Baseline | 0.9138 | 0.7013 | 0.1743 |
| AugMix | **0.9228** | **0.8302** | 0.0251 |
| Student | 0.8900 | 0.6845 | 0.0742 |

# 🧠 Key Insights

### 1. Robustness is Multi-Dimensional
- Corruption robustness ≠ Adversarial robustness


### 2. AugMix is Highly Effective for CIFAR-10-C
- +13% Mean Corruption Accuracy (mCA) improvement
- Especially strong for noise


### 3. Adversarial Vulnerability Remains Severe
- PGD drastically reduces accuracy
- L∞ attacks are the most destructive


### 4. Distillation Does Not Transfer Robustness Well
- Student fails to inherit teacher robustness
- Suggests robustness is not easily compressible
