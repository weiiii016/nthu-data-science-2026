# Data Science 課程作業總覽

> 國立清華大學 資訊工程學系開設課程 · 2026 Spring（大三下）
> 實作 **Data Mining、Supervised Learning、Black-box Optimization、Model Compression** 四個主題。

| # | 主題 | Core Method | Language / Tools | 成果 |
|---|------|-------------|------------------|------|
| HW1 | Frequent Pattern Mining | FP-Growth（from scratch） | C++17 | **95 / 100**：5 筆 test cases 全數通過，大型 test case 的 runtime 排名進入中段 |
| HW2 | Classification | XGBoost + HistGradientBoosting Ensemble、Threshold Optimization | Python, scikit-learn, XGBoost | **80 / 100**：Public Macro-F1 **0.8143**，Private 0.7847，兩者皆超越 Random Forest / XGBoost baseline |
| HW3 | Black-box Global Optimization | CMA-ES + Random Warm-up + Stagnation Restart | Python（NumPy only） | **86.5 / 100**：8 個 functions 全數優於 PSO，4 個優於 advanced baseline，其中 1 個找到 global minimum |
| HW4 | Deep Model Compression | Pruning → K-means Weight Sharing → Huffman Coding | PyTorch, scikit-learn | AlexNet / CIFAR-10 Compression Rate **269.73×**，Accuracy 維持 59.99%；**100 / 100**，Compression Rate 排名全班前 1/5 |

---

## HW1 — Frequent Pattern Mining（FP-Growth）

### 問題
給定 transaction database 與 minimum support，找出所有 frequent itemsets 並輸出其 support。
- 每筆 transaction 最多 200 個 items，item 範圍 0–999，最多 100,000 筆 transactions
- **禁止**使用任何 frequent pattern 相關 library；最後一筆 test case 依 runtime 排名給分

### 做法（[`HW1/v2.cpp`](HW1/v2.cpp)）
以 C++ from scratch 實作 **FP-Growth**：

1. **Fast I/O**：逐字元 parse 數字（不使用 `stringstream`），每筆 transaction 內的 items 排序並去重。
2. **FP-Tree Construction**：
   - 先統計 global frequency，過濾 infrequent items，依 frequency 遞減排序後插入 tree。
   - **Header table** 以 `head / tail` pointer 維護同一 item 的 node-link（linked list），insertion 為 O(1)。
   - 因 item 範圍已知（0–999），header table 與 frequency table 使用 fixed-size array 而非 hash map，降低常數。
3. **Recursive Mining**：
   - 由 frequency 最低的 item 開始，沿 node-link 收集 **conditional pattern base**，建立 **conditional FP-tree** 後 recursion。
   - **Single-path Optimization**：若 tree 退化為 single path，直接以 bitmask 列舉所有 subsets，support 取 path 上最小 count，省去後續 recursion。
4. **Output**：排序去重後輸出 `pattern:support`（rounding 至小數第 4 位）。

```bash
g++ -std=c++2a -O2 -o hw1 v2.cpp
./hw1 0.2 sample.txt sample_out.txt
```

### 結果
- 成績 **95 / 100**
- 5 筆 test cases（最大一筆約 10 萬筆 transactions）**全數正確通過**；最後一筆依 runtime 分三段計分，本作業落在中段。

### 學到的重點
- Algorithm 層面（FP-Growth vs. Apriori，避免 candidate explosion）與 implementation 層面（memory allocation、data structure 選擇）都會顯著影響效能。
- 利用問題限制（item 範圍有界）換取更快的 data structure。

---

## HW2 — Bankruptcy Prediction（Kaggle Classification Competition）

### 問題
以 Taiwan Economic Journal (TEJ) 衍生資料進行 **binary classification**，預測公司是否破產。
- Training set 5,543 筆、testing set 1,386 筆，95 個 features（numeric + nominal）
- **Highly imbalanced**：positive class 僅 4.76%；資料含 **missing values 與 random noise**
- Evaluation metric：**Macro-F1**（以 Private Leaderboard 計分）

### 做法（[`HW2/112020016_DS_hw2.ipynb`](HW2/112020016_DS_hw2.ipynb)）
迭代式地建立並改進 pipeline：

| Stage | 內容 |
|-------|------|
| Preprocessing | Median imputation、**missing indicator** features；以 training set 的 quantile 做 **quantile clipping**（0.5%/99.5%、1%/99%）抑制 noise 與 outliers |
| Class Imbalance | XGBoost `scale_pos_weight ≈ 20`、HGB 使用 fold-specific sample weight |
| Models | **XGBoost** 與 **HistGradientBoosting**，5-fold Stratified Cross-Validation |
| Hyperparameter Tuning | 小範圍 grid sweep（learning rate、max depth、regularization 等），以 OOF Macro-F1 選模型 |
| Threshold Tuning | 因 metric 為 Macro-F1，不使用預設 threshold 0.5，而以 OOF predictions 做 **two-stage threshold search（coarse 0.01 → fine 0.001）** |
| Ensemble | **Probability-weighted** 與 **rank-based** 兩種 ensemble，在 OOF 上同時搜尋 weights 與 threshold |

### 結果

| Method | Public | Private |
|--------|-------:|--------:|
| Baseline – Random Forest | 0.64825 | 0.62444 |
| Baseline – XGBoost | 0.75427 | 0.76849 |
| **本作業** | **0.8143** | **0.7847** |

OOF Macro-F1 由 single XGBoost 的 0.778 提升到 clipping + ensemble 後約 0.79。

### 學到的重點
- 在 imbalanced data 上，**decision threshold** 與 model 本身一樣重要；evaluation metric 決定 optimization objective。
- 以 OOF predictions 做所有選擇（threshold、ensemble weights），避免 overfit 到 public leaderboard。

---

## HW3 — Black-box Global Optimization（CMA-ES）

### 問題
給定 4 個未知的 objective functions（只能透過 `evaluate(x)` 取得 function value），在**有限的 function evaluations**（1,000 / 1,500 / 2,000 / 2,500 次）內找出 global minimum。
- 僅允許使用 **NumPy**，不得呼叫現成的 optimization functions
- 與 Random Search、PSO，以及 DIRECT-L / CoDE / EDA-LS 等 baselines 比較，另有 4 個 private (hidden) functions

### 做法（[`HW3/112020016_hw3.py`](HW3/112020016_hw3.py)）
完整手刻 **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**，並加入兩項改良：

1. **Phase 1 — Random Warm-up**：使用 2% budget（5–50 個點）做 uniform sampling，作為 CMA-ES 的 initial mean。
2. **Phase 2 — CMA-ES Core**：
   - Standard parameter setting：λ = 4 + ⌊3 ln n⌋、μ = λ/2、log-scaled recombination weights、`cc / cs / c1 / cμ / damps`
   - **Cumulative Step-size Adaptation (CSA)**（evolution path `p_σ`）與 **rank-one + rank-μ covariance update**
   - 週期性 **eigendecomposition** `C = B D² Bᵀ`，並強制 symmetric 以確保 numerical stability
   - Sampled points clip 至 search domain 邊界
3. **Stagnation Restart**：連續 30 generations 無改善即 restart，以目前 best solution 加上遞減 noise 為新 mean，σ 逐次減半，聚焦於已知的 promising region；σ 正常收斂至 1e-12 則不觸發 restart，避免浪費 budget。

### 結果
成績 **86.5 / 100**（4 個 public functions 各佔 15 分，4 個 private functions 各佔 10 分）。

| Function | f1 | f2 | f3 | f4 |
|----------|---:|---:|---:|---:|
| Public | 75% | **95%** | **95%** | **95%** |
| Private | 75% | **100%** | 75% | 75% |

Grading：75% 代表結果介於 PSO 和 advanced baseline 之間，且在班上前 1/2；95% 代表優於 DIRECT-L / CoDE / EDA-LS 等 advanced baselines，且在班上前 1/4；100% 代表**找到 global minimum**。
- 8 個 functions **全部優於 Random Search 與 PSO baseline**
- Public f2–f4 優於 advanced baselines，並排名班上前 1/4
- Private f2 **找到 global minimum**

### 學到的重點
- 在極少 function evaluations 下，**exploration / exploitation** 的平衡比 algorithm 本身更關鍵。
- 讀 paper 並把 evolution strategy 的數學式正確轉成程式（包含 numerical stability 處理）。

---

## HW4 — Deep Compression

### 問題
參考 Han et al., *Deep Compression* ([arXiv:1510.00149](https://arxiv.org/abs/1510.00149))，在 **CIFAR-10** 上壓縮 **AlexNet**（約 20.34M parameters），條件為 Accuracy > 58%，以 **Compression Rate** 全班排名。

### 做法（[`HW4/hw4/`](HW4/hw4/)）
完成三階段 pipeline：

```
Pruning ──► Retrain ──► Weight Sharing (K-means) ──► Huffman Coding
```

1. **Pruning**（[`net/prune.py`](HW4/hw4/net/prune.py)、[`pruning.py`](HW4/hw4/pruning.py)）
   - 實作 `prune_by_percentile`（每層依 percentile 決定 threshold）與 `prune_by_std`（擴展至 conv layers）
   - **Retrain 時將 pruned weights 的 gradient 歸零**，確保 sparsity pattern 不被破壞
2. **Weight Sharing / Quantization**（[`net/quantization.py`](HW4/hw4/net/quantization.py)）
   - 對 conv layers 的 non-zero weights 做 **K-means clustering（5 bits = 32 clusters，linear initialization）**，以 centroid 取代原值，並保持 zero weights 的位置不變
3. **Huffman Coding**（[`net/huffmancoding.py`](HW4/hw4/net/huffmancoding.py)）
   - 將 4D conv weights 拆成 Kn×Ch 個 2D CSR matrices，串接 data / indices / indptr 後分別做 Huffman encoding 與 decoding

### Layer-wise Pruning Rate Tuning
FC layers 佔絕大多數 parameters（fc2 即 16.77M），因此對 fc2 的 pruning 最為激進；conv1 對 accuracy 最敏感，保留較多 weights。透過多組實驗尋找 accuracy 與 compression rate 的平衡：

| Exp. | conv1 | conv2 | conv3 | conv4 | conv5 | fc1 | fc2 | fc3 | Acc. after Pruning | Acc. after Retraining | Final Acc. | Compression Rate |
|------|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| A | 20% | 10% | 7% | 4.5% | 4% | 3.5% | 0.6% | 10% | 41.22% | 60.33% | 60.35% | 167.54× |
| B | 20% | 10% | 6% | 4% | 4% | 3.5% | 0.4% | 20% | 41.48% | 60.33% | 59.90% | 185.71× |
| C | 30% | 8% | 6% | 4% | 4% | 3% | 0.3% | 30% | 45.72% | 60.18% | 60.23% | 198× |
| **D（submitted）** | **50%** | **8%** | **4%** | **2%** | **4%** | **2%** | **0.05%** | **20%** | 51.31% | 60.30% | **59.99%** | **269.73×** |

*（表中百分比為各層保留的 weights 比例；original model accuracy 61.57%）*

最終 model prune 掉 **99.35%** weights（pruning 階段 152.7×），再經 quantization 與 Huffman coding 後達到 **269.73× compression rate**，accuracy 僅下降約 1.6 個百分點。

### 成績
**100 / 100**：Pruning、Quantization、Huffman Coding 三個部分都完成，compression rate 排名進入**全班前 1/5**，拿到該項滿分。

### 學到的重點
- 不同 layer 對 pruning 的 sensitivity 差異很大，**layer-wise tuning** 比單一 global ratio 有效得多。
- Retraining 能大幅恢復 pruning 造成的 accuracy loss（41% → 60%）。

---

## 技能總結

- **Algorithm Implementation**：FP-Growth、CMA-ES、K-means Weight Sharing、Huffman Coding，皆依 paper 或課程內容自行實作
- **Machine Learning Practice**：Class Imbalance Handling、Cross-Validation、Threshold Optimization、Ensemble Learning
- **Deep Learning**：PyTorch training pipeline、Model Pruning 與 Compression
- **Performance-oriented Programming**：C++ data structure 與 memory allocation optimization
- **Experimental Methodology**：系統性的 hyperparameter sweep 與結果比較
