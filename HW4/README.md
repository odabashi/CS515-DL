# 📈 Financial Forecasting with LSTM and GRU

This project evaluates recurrent neural networks for stock return forecasting using daily OHLC data from Yahoo Finance.

We study three tasks:

* **Part B:** Exact d-day return ratio forecasting
* **Part C:** Rolling-average return forecasting
* **Part D:** Turning-point detection with buy/pass signals

# 📌 Dataset

* **Tickers:** AAPL, NVDA, MSFT, META, AMZN, TSLA, GOOGL
* **Period:** January 2020 – December 2025
* **Features:** Open, High, Low, Close
* **Lookback window:** $T = 20$
* **Forecast horizons:** $d = 1, \dots, 5$

## 🔹 Chronological Split

| Split      | Period              | Samples |
| ---------- | ------------------- | ------: |
| Train      | Jan 2020 – Jul 2024 |   7,896 |
| Validation | Aug 2024 – Dec 2024 |     574 |
| Test       | Jan 2025 – Dec 2025 |   1,575 |

Each stock was normalized independently using min-max scaling fitted only on the training split to avoid data leakage.

# 📌 Part B — Exact Return Ratio Forecasting

* **Models:** StockLSTM, StockGRU
* **Target:** Exact future return ratio
* **Loss:** $MSE$
* **Optimizer:** AdamW
* **Auxiliary Features:** Learned moving-average channels using 1D convolution

## 🔹 Overall Test Results

| Model     |          MSE |         RMSE |          MAE | Parameters |
| --------- | -----------: | -----------: | -----------: | ---------: |
| StockLSTM |     0.001925 |     0.042560 |     0.030345 |      1,837 |
| StockGRU  | **0.001915** | **0.042468** | **0.030210** |      1,421 |

## 🔹 Per-Horizon RMSE

| Horizon |   StockLSTM |    StockGRU |
| ------- | ----------: | ----------: |
| d=1     |     0.02599 | **0.02598** |
| d=2     |     0.03572 | **0.03560** |
| d=3     | **0.04430** |     0.04481 |
| d=4     |     0.05124 | **0.05044** |
| d=5     |     0.05555 | **0.05551** |

## 🔍 Observations

* LSTM and GRU performed almost identically.
* GRU achieved slightly lower RMSE while using about **23% fewer parameters**.
* Error increased as the forecasting horizon became longer.
* Both models converged early before the 150-epoch maximum.

## 🖼️ Figures

* [Part B Per-Horizon RMSE](./images/part_b_per_horizon_rmse.png)
* [LSTM Exact Dashboard](./images/lstm_exact_dashboard.png)
* [GRU Exact Dashboard](./images/gru_exact_dashboard.png)

👉 **Conclusion:** GRU provides nearly the same forecasting accuracy as LSTM with fewer parameters.

# 📌 Part C — Rolling-Average Return Forecasting

* **Models:** StockLSTM, StockGRU
* **Target:** Weighted rolling-average future return
* **Rolling window:** $l = 3$

## 🔹 Overall Test Results

| Model     |      MSE |         RMSE |          MAE | Parameters |
| --------- | -------: | -----------: | -----------: | ---------: |
| StockLSTM | 0.001098 |     0.030681 |     0.021754 |      1,837 |
| StockGRU  | 0.001104 | **0.030466** | **0.021614** |      1,421 |

## 🔹 Exact vs Rolling RMSE

| Model     | Exact RMSE | Rolling RMSE | Improvement |
| --------- | ---------: | -----------: | ----------: |
| StockLSTM |    0.04256 |      0.03068 |        ~28% |
| StockGRU  |    0.04247 |      0.03047 |        ~28% |

## 🔍 Observations

* Rolling-average targets significantly reduced forecasting error.
* RMSE dropped from about **0.0425** to **0.0305**.
* Training became smoother and converged earlier.
* Target smoothing had a much larger impact than choosing LSTM vs GRU.

## 🖼️ Figures

* [Exact vs Rolling RMSE](./images/part_c_exact_vs_rolling.png)
* [Part C Per-Horizon RMSE](./images/part_c_per_horizon_rmse.png)
* [LSTM Rolling Dashboard](./images/lstm_rolling_dashboard.png)
* [GRU Rolling Dashboard](./images/gru_rolling_dashboard.png)

👉 **Conclusion:** Rolling-average forecasting is more stable and more accurate than exact-return forecasting.


# 📌 Part D — Turning-Point Detection

- **Models:** BiLSTM, BiGRU
- **Task:** Predict buy/pass signal
- **Target:** Gross return using max future price
- **Threshold:** $γ = 1.1$
- **Loss:** $MSE + λ \cdot BCE$

## 🔹 Why Gross Return Was Used

The assignment defines the turning-point target using net return:

$
\frac{P_{new} - P_{old}}{P_{old}}
$

With $γ = 1.1$, this would require a **110% price increase within 5 trading days**, which is unrealistic for large-cap stocks and produces almost all PASS labels.

Therefore, the implementation used gross return:

$
\frac{P_{new}}{P_{old}}
$

Under this interpretation, $γ = 1.1$ means a **10% price increase**, which is more realistic.


## 🔹 Main Test Results

| Metric     |   BiLSTM |        BiGRU |
| ---------- | -------: | -----------: |
| MSE        | 0.003198 | **0.001876** |
| RMSE       | 0.052597 | **0.042569** |
| MAE        | 0.040478 | **0.030422** |
| Accuracy   |   0.9206 |       0.9206 |
| Precision  |   0.0000 |       0.0000 |
| Recall     |   0.0000 |       0.0000 |
| F1         |   0.0000 |       0.0000 |
| Parameters |    3,614 |        2,782 |

## 🔹 Per-Horizon RMSE

| Horizon |      BiLSTM |       BiGRU |
| ------- | ----------: | ----------: |
| d=1     | **0.02646** |     0.03417 |
| d=2     |     0.03553 | **0.03383** |
| d=3     |     0.05391 | **0.04131** |
| d=4     |     0.08584 | **0.05054** |
| d=5     |     0.06124 | **0.05300** |

## 🔍 Observations

* BiGRU achieved better overall regression performance.
* BiLSTM performed slightly better only at d=1.
* BiLSTM error increased sharply at longer horizons, especially d=4.
* Both classifiers predicted every test sample as PASS.
* Accuracy was high only because the dataset was imbalanced.

## 🔹 Confusion Matrix

| Actual / Predicted | Pass | Buy |
| ------------------ | ---: | --: |
| Actual Pass        | 1450 |   0 |
| Actual Buy         |  125 |   0 |

👉 **Conclusion:** The regression head learned useful return patterns, but the classification head collapsed to the majority PASS class.

## 🖼️ Figures

* [Part D Per-Horizon RMSE](./images/part_d_per_horizon_rmse.png)
* [BiLSTM Confusion Matrix](./images/bilstm_turning_point_confusion_matrix.png)
* [BiGRU Confusion Matrix](./images/bigru_turning_point_confusion_matrix.png)
* [BiLSTM Dashboard](./images/bilstm_turning_point_dashboard.png)
* [BiGRU Dashboard](./images/bigru_turning_point_dashboard.png)


# 📌 Part D Ablation Study — Class Imbalance

To investigate the classifier collapse, three additional experiments were performed:

1. Increase BCE weight: $λ = 5$
2. Use positive-class weighting: $pos\_weight = 10$
3. Combine both methods

## 🔹 Ablation Results

| Configuration          | Model  |       RMSE |   Accuracy |  Precision | Recall |     F1 | TP | FP |
|------------------------| ------ | ---------: | ---------: | ---------: | -----: | -----: | -: | -: |
| Baseline ($λ=0.5$)     | BiLSTM |     0.0526 |     0.9206 |     0.0000 | 0.0000 | 0.0000 |  0 |  0 |
| Baseline ($λ=0.5$)     | BiGRU  | **0.0426** |     0.9206 |     0.0000 | 0.0000 | 0.0000 |  0 |  0 |
| $λ=5$                  | BiLSTM |     0.0755 |     0.9206 |     0.0000 | 0.0000 | 0.0000 |  0 |  0 |
| $λ=5$                  | BiGRU  |     0.0918 |     0.9206 |     0.0000 | 0.0000 | 0.0000 |  0 |  0 |
| $pos\_weight=10$        | BiLSTM |     0.0493 |     0.9206 |     0.0000 | 0.0000 | 0.0000 |  0 |  0 |
| $pos\_weight=10$        | BiGRU  |     0.0976 |     0.9206 |     0.0000 | 0.0000 | 0.0000 |  0 |  0 |
| $λ=5 + pos\_weight=10$ | BiLSTM |     0.1970 |     0.9206 |     0.0000 | 0.0000 | 0.0000 |  0 |  0 |
| $λ=5 + pos\_weight=10$ | BiGRU  |     0.0838 | **0.9219** | **1.0000** | 0.0160 | 0.0315 |  2 |  0 |

## 🔍 Ablation Observations

### 1. Increasing BCE Weight Alone

* Did not improve classification.
* Both models still predicted all samples as PASS.
* Regression performance became worse.

👉 Stronger BCE weighting disrupted regression but did not solve class imbalance.

### 2. Positive-Class Weighting

* BUY probabilities increased from about 0.07–0.08 to around 0.39–0.46.
* However, probabilities still stayed below the 0.5 threshold.
* Precision, recall, and F1 remained zero.

👉 The model reacted to the loss change but still failed to produce BUY predictions.

### 3. BCE Weight + Positive-Class Weight

* Only this setting produced BUY predictions.
* BiGRU detected 2 true BUY samples.
* Precision became 1.0 because there were no false positives.
* Recall remained extremely low at 0.016.

👉 The classifier became very conservative: it only predicted BUY in rare cases.

### 4. Regression Trade-Off

* Stronger classification weighting damaged regression performance.
* This shows negative transfer between regression and classification objectives.

👉 The shared encoder struggled to optimize both tasks simultaneously.


# 📊 Final Summary

| Model / Experiment |      MSE |         RMSE |          MAE | Accuracy | Precision | Recall |     F1 | Parameters |
| ------------------ | -------: | -----------: | -----------: | -------: | --------: | -----: | -----: | ---------: |
| LSTM Exact         | 0.001925 |     0.042560 |     0.030345 |        - |         - |      - |      - |      1,837 |
| GRU Exact          | 0.001915 |     0.042468 |     0.030210 |        - |         - |      - |      - |      1,421 |
| LSTM Rolling       | 0.001098 |     0.030681 |     0.021754 |        - |         - |      - |      - |      1,837 |
| GRU Rolling        | 0.001104 | **0.030466** | **0.021614** |        - |         - |      - |      - |      1,421 |
| BiLSTM Turning     | 0.003198 |     0.052597 |     0.040478 |   0.9206 |    0.0000 | 0.0000 | 0.0000 |      3,614 |
| BiGRU Turning      | 0.001876 |     0.042569 |     0.030422 |   0.9206 |    0.0000 | 0.0000 | 0.0000 |      2,782 |

---

# 🧠 Key Insights

## 1. GRU Is More Parameter-Efficient

GRU achieved similar or slightly better forecasting performance than LSTM while using fewer parameters.

## 2. Target Design Matters More Than Architecture

Rolling-average targets reduced RMSE by about **28%**, which is much larger than the difference between LSTM and GRU.

## 3. Forecasting Error Increases with Horizon

All models performed best at d=1 and worse at longer horizons.

## 4. Turning-Point Detection Is Harder Than Regression

The regression head learned useful patterns, but the classifier collapsed to the majority PASS class.

## 5. Accuracy Is Misleading Under Class Imbalance

The turning-point classifiers achieved 92% accuracy while detecting zero BUY signals.

## 6. Loss Reweighting Alone Is Not Enough

Increasing BCE weight and adding positive-class weighting increased BUY probabilities, but did not reliably solve the classification problem.

## 7. Future Improvements

Possible improvements include:

* More informative features such as volume, volatility, technical indicators, or news sentiment
* Focal loss
* Oversampling BUY samples
* Lower or adaptive decision threshold
* Separate classifier instead of multi-task learning
* Transformer-based time-series models

# ✅ Final Conclusion

Recurrent neural networks can learn meaningful short-term stock-return patterns from OHLC data. GRU models are especially attractive because they achieve similar or better results than LSTM models with fewer parameters.

The most important result is that **rolling-average return forecasting significantly improves performance**, showing that smoothing noisy financial targets can make the prediction task more stable.

However, turning-point detection remains challenging. The buy/pass classifier suffered from severe class imbalance and mostly learned the majority PASS class. The ablation study showed that simple loss reweighting is not sufficient; richer features and more specialized imbalance-handling methods are needed for reliable trading-signal generation.
