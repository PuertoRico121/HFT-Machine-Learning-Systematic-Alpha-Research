# Machine Learning for Limit Order Book Signals and Systematic Alpha Research

## Overview

This project studies whether high-frequency trade and quote data contain exploitable information about short-horizon equity price movements, and whether machine-learning prediction probabilities can be transformed into economically meaningful systematic alpha signals.

The research starts with a **market-microstructure classification problem**: predicting whether the mid-price will increase at the next market event using engineered features from millisecond-level Trade and Quote (TAQ) data.

It then extends the prediction framework into a systematic alpha research pipeline:

```text
Raw TAQ Data
    ↓
Trade / Quote Timestamp Alignment
    ↓
Microstructure Feature Engineering
    ↓
Next-Event Price Direction Classification
    ↓
Probability-Based Alpha Scores
    ↓
Quintile Sorting & Rank IC
    ↓
Signal Persistence / Alpha Decay
    ↓
Extreme-Score Long / Short Signals
    ↓
Cost-Aware Out-of-Sample Backtest
    ↓
NAV, Sharpe, Drawdown & Cost Sensitivity
```

The objective is therefore broader than maximizing classification accuracy. The project separates three questions:

1. **Predictive skill** — can microstructure features predict short-horizon price direction?
2. **Alpha persistence** — does the model score rank future returns, and how quickly does that information decay?
3. **Trading economics** — does the signal remain economically meaningful after signal selection, execution timing and transaction costs?

---

## Research Problem

At every trade event \(t\), the model observes the latest available market state and predicts whether the next mid-price will be higher than the current mid-price:

\[
y_t =
\begin{cases}
1, & Mid_{t+1} > Mid_t \\
0, & \text{otherwise}
\end{cases}
\]

where:

\[
Mid_t = \frac{Ask_t + Bid_t}{2}
\]

Rather than treating the model only as a hard binary classifier, the estimated probability

\[
P(y_t=1 \mid X_t)
\]

is subsequently used as a **continuous alpha score**.

This allows the project to study not only whether the model predicts direction correctly, but whether higher model scores systematically correspond to stronger subsequent returns.

---

# 1. Data

## Source

The project uses millisecond-level **TAQ (Trade and Quote)** data containing:

### Trade data

- Timestamp
- Stock symbol
- Trade price
- Trade size

### Quote data

- Timestamp
- Stock symbol
- Best bid
- Best ask
- Bid size
- Ask size

The analysis focuses on the high-activity opening-hour window:

```text
09:30:00 – 10:30:00
```

The alpha-research extension uses:

| Sample | Period | Observations |
|---|---|---:|
| Training | 12 March 2024 | 51,053 |
| Strict OOS | 11 July 2024 | 44,782 |
| Symbols | — | 11 |

July is kept strictly out of sample for the alpha-testing stage.

---

# 2. Trade–Quote Alignment

Trades and quotes arrive as separate event streams, so the first task is to reconstruct the market state observable when each trade occurs.

For each trade, the project uses a backward as-of join:

```python
pd.merge_asof(
    trades,
    quotes,
    on="TIMESTAMP",
    by="SYM_ROOT",
    direction="backward"
)
```

Each trade is therefore paired with the **latest quote available at or before its timestamp**.

This produces an event-level dataset where each row contains:

```text
Trade Event
+ Latest Observable Bid / Ask
+ Order-Book Depth
+ Engineered Microstructure Features
```

The alpha extension additionally retains one minute of pre-open quote history so trades immediately after 09:30 can still be matched to the latest observable quote.

---

# 3. Microstructure Feature Engineering

The models are trained on features designed to represent liquidity, order-book pressure, trade direction and short-term price dynamics.

## 3.1 Normalized Spread

\[
Spread =
\frac{Ask-Bid}{Mid}
\]

Captures the width of the quoted market relative to the current price level.

---

## 3.2 Volume Imbalance

\[
VolumeImbalance =
\frac{BidSize-AskSize}
{BidSize+AskSize}
\]

Measures asymmetry between displayed bid and ask liquidity.

Positive values indicate relatively greater bid-side depth, while negative values indicate greater ask-side depth.

---

## 3.3 Microprice

A depth-weighted price estimate is constructed as:

\[
Microprice =
\frac{Ask \times BidSize + Bid \times AskSize}
{BidSize+AskSize}
\]

The alpha extension expresses this relative to the current mid-price:

\[
MicropriceDisp =
\frac{Microprice-Mid}{Mid}
\]

This captures whether order-book depth shifts the implied price above or below the quoted midpoint.

---

## 3.4 Trade Location / Trade Pressure

The transaction price is compared with the prevailing midpoint and quoted spread.

The alpha-research version uses:

\[
TradeLocation =
\frac{TradePrice-Mid}
{(Ask-Bid)/2}
\]

with extreme observations clipped to:

\[
[-2,2]
\]

This provides a standardized measure of whether transactions occur toward the bid or ask side of the market.

---

## 3.5 Signed Volume

The original classification study constructs directional signed volume from trade direction and trade size.

For the alpha extension, trade size is log-compressed:

\[
SignedLogVolume =
TradeSign \times \log(1+Size)
\]

where the trade sign is determined from the transaction price relative to the current mid-price.

---

## 3.6 Short-Term Momentum

Event-level momentum is calculated from consecutive mid-prices within each symbol:

\[
Momentum_t =
\frac{Mid_t}{Mid_{t-1}}-1
\]

This captures the most recent direction of local price movement.

---

# 4. Machine-Learning Models

The original prediction framework compares three supervised-learning models with different levels of model capacity.

## Logistic Regression

Logistic Regression serves as the interpretable linear baseline.

The model uses:

- Standardized microstructure features
- L2 regularization
- Time-series-aware train/test splitting

Average classification performance:

| Metric | Result |
|---|---:|
| Train Accuracy | 68.59% |
| Test Accuracy | 71.91% |

Coefficient analysis identifies **trade pressure** as the strongest linear predictor.

---

## Random Forest

Random Forest is used to capture nonlinear relationships and conditional interactions among microstructure variables.

Key parameters include:

```python
n_estimators = 100
max_depth = 6
min_samples_leaf = 100
bootstrap = True
```

Performance:

| Metric | Result |
|---|---:|
| Train Accuracy | 71.53% |
| Test Accuracy | 74.04% |
| Train ROC-AUC | ~0.73 |
| Test ROC-AUC | ~0.74 |

Random-Forest feature importance shows that most predictive information is concentrated in a small set of order-flow variables:

| Feature | Importance |
|---|---:|
| Trade Pressure | ~38.5% |
| Signed Volume | ~24.7% |
| Momentum | ~22.8% |
| Spread | ~6.9% |
| Smart Price | ~5.6% |
| Volume Imbalance | ~1.5% |

Trade pressure, signed volume and momentum together account for roughly **86% of total feature importance**.

---

## Feedforward Neural Network

A multilayer perceptron is used as a flexible nonlinear classifier.

Architecture:

```text
Input Features
    ↓
Dense(25) + ReLU
    ↓
Dense(10) + ReLU
    ↓
Dense Hidden Layer + ReLU
    ↓
Sigmoid Output
```

Training uses:

- Adam optimizer
- Binary cross-entropy loss
- L2 regularization
- Mini-batch training

The output is a probability of the next mid-price moving upward.

The model achieves approximately:

```text
Test Accuracy: ~72.6%
ROC-AUC:       ~0.70–0.71
```

Its probability output also enables threshold-based analysis of the trade-off between overall accuracy and minority-class detection.

---

# 5. Classification Threshold Analysis

Because next-event upward moves represent the minority class, overall accuracy alone can hide weak sensitivity to positive events.

The project therefore evaluates confusion matrices, recall and F1-score under different decision thresholds.

For example, on the later-period Neural Network evaluation, lowering the classification threshold from:

```text
0.50 → 0.40
```

substantially increases upward-move recall, while reducing overall accuracy.

Random Forest shows an even stronger trade-off.

At a 0.5 threshold:

```text
Class-1 Recall: 13.61%
F1:             0.220
Accuracy:       72.85%
```

At a 0.4 threshold:

```text
Class-1 Recall: 68.44%
F1:             0.516
Accuracy:       63.86%
```

This motivates the next stage of the project: retaining the **continuous probability score** rather than immediately reducing model output to a binary prediction.

---

# 6. Leakage-Safe Alpha Research Extension

The alpha-research section rebuilds a compact dataset directly from raw TAQ files.

The design follows a strict chronological separation:

```text
March 12, 2024
    ↓
Feature preprocessing
Model fitting
Score distribution
Trading thresholds

July 11, 2024
    ↓
Frozen preprocessing
Frozen models
Frozen thresholds
Strict OOS evaluation
```

Only information observable at the signal timestamp is used as input.

Future prices are created only after feature construction and are used exclusively for labels, forward-return analysis and backtesting.

A fixed liquidity screen is applied:

\[
NormalizedSpread \le 0.002
\]

equivalent to a maximum normalized quoted spread of approximately 20 bps.

---

# 7. Predictive Models as Alpha Scorers

The alpha extension focuses on:

- Logistic Regression
- Random Forest

The models are trained on March data and applied without refitting to July.

### ROC-AUC

| Model | March Train AUC | July OOS AUC |
|---|---:|---:|
| Logistic Regression | 0.6734 | 0.6624 |
| Random Forest | 0.7113 | **0.6865** |

Random Forest therefore provides the stronger nonlinear ranking signal in the strict OOS sample and is used as the primary alpha score for the subsequent strategy research.

---

# 8. Prediction Probability → Alpha Score

Instead of converting Random-Forest probabilities immediately into:

```text
0 = Short / No Up Move
1 = Long / Up Move
```

the probability itself is treated as a **continuous alpha score**.

July observations are sorted into five equal-frequency score buckets.

Average subsequent **5-event mid-price returns** are then calculated for each quintile.

| RF Signal Quintile | Average 5-Event Forward Return |
|---|---:|
| Q1 | -2.673 bps |
| Q2 | -2.337 bps |
| Q3 | -2.632 bps |
| Q4 | -1.713 bps |
| Q5 | -0.767 bps |

The extreme-bucket spread is:

\[
Q5-Q1
=
-0.767-(-2.673)
=
\mathbf{1.906\ bps}
\]

Although the intermediate buckets are not perfectly monotonic, the highest-scoring observations have materially stronger subsequent returns than the lowest-scoring observations.

This provides an economic interpretation of the classifier probability:

> **The model probability contains short-horizon return-ranking information beyond its use as a binary classification score.**

---

# 9. Signal Persistence and Alpha Decay

A useful microstructure signal may have predictive information immediately after formation but lose relevance rapidly as new trades and quotes arrive.

The project therefore measures **signal persistence** using Spearman Rank Information Coefficient:

\[
IC_h =
Corr_{rank}
(
AlphaScore_t,
Return_{t+h\rightarrow t+h+1}
)
\]

The analysis measures the relationship between the current alpha score and a single-event return occurring after lags of:

```text
1, 3, 5 and 10 market events
```

### OOS Rank IC

| Model | 1 Event | 3 Events | 5 Events | 10 Events |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.0816 | 0.0218 | 0.0123 | 0.0011 |
| Random Forest | **0.0857** | 0.0204 | 0.0123 | -0.0009 |

The pattern shows rapid alpha decay:

```text
Signal Event
    ↓
1 event   → RF Rank IC = 0.0857
    ↓
3 events  → RF Rank IC = 0.0204
    ↓
5 events  → RF Rank IC = 0.0123
    ↓
10 events → RF Rank IC ≈ 0
```

The strongest predictive information therefore exists immediately following the signal and largely disappears within approximately ten subsequent market events.

---

# 10. Systematic Long / Short Signal Construction

The trading rule uses the **Random-Forest probability** as its alpha score.

Trading thresholds are determined exclusively from the March training-score distribution:

```text
Bottom 5% of scores → Short
Top 5% of scores    → Long
Otherwise           → No Position
```

The resulting March thresholds are:

```text
Lower threshold: 0.0978
Upper threshold: 0.5098
```

These thresholds are then frozen and applied directly to July OOS observations.

This creates a selective strategy that trades only the most extreme model signals.

---

# 11. Backtest Timing

For a signal generated at market event \(t\):

```text
Signal observed at t
        ↓
Enter at t+1
        ↓
Hold for 10 subsequent events
        ↓
Exit
```

Additional rules include:

- Entry must occur strictly after signal generation.
- Trades within the same symbol cannot overlap.
- Mid-prices are used for signal-research execution.
- The same rules are applied across the OOS sample.
- A fixed implementation cost is deducted from each completed trade.

The baseline round-trip cost assumption is:

\[
0.5\text{ bp}
\]

---

# 12. OOS Trading Results

Applying the frozen Random-Forest model and March-derived thresholds to July produces:

```text
Total OOS Trades: 2,322
Long Trades:      1,524
Short Trades:       798
```

### Per-Trade Performance

| Metric | Result |
|---|---:|
| Average Gross Return | **1.561 bps** |
| Average Net Return | **1.061 bps** |
| Gross Hit Rate | 54.48% |
| Net Hit Rate | 52.20% |
| Gross Per-Trade Sharpe | 0.028 |
| Net Per-Trade Sharpe | 0.019 |

The reported Sharpe metric is:

\[
Sharpe_{trade}
=
\frac{Mean(Return_{trade})}
{Std(Return_{trade})}
\]

and is therefore a **per-trade mean-to-volatility ratio**, not an annualized portfolio Sharpe ratio.

---

# 13. OOS NAV Construction

Individual event-level trades are not compounded as separate full-capital portfolios.

Instead, trades entering within the same minute are first aggregated using equal weights:

```text
Individual Trade Returns
        ↓
1-Minute Equal-Weighted Return
        ↓
Chronological Compounding
        ↓
Portfolio NAV
```

The resulting July OOS performance is:

| Metric | Result |
|---|---:|
| Gross Ending NAV | 1.006704 |
| Net Ending NAV | 1.003689 |
| Gross Total Return | **+0.6704%** |
| Net Total Return | **+0.3689%** |
| Maximum Net Drawdown | **-0.72%** |

The baseline net result assumes a **0.5 bp round-trip implementation cost**.

---

# 14. Transaction-Cost Sensitivity

Because the alpha signal operates over a very short event horizon, implementation costs materially affect economic performance.

The same OOS trades are therefore revalued under multiple cost assumptions without refitting the model or changing signal thresholds.

| Round-Trip Cost | Ending NAV | Total OOS Return |
|---:|---:|---:|
| 0.0 bp | 1.006704 | **+0.6704%** |
| 0.5 bp | 1.003689 | **+0.3689%** |
| 1.0 bp | 1.000683 | **+0.0683%** |
| 2.0 bp | 0.994696 | **-0.5304%** |

The cost analysis highlights the relationship between short-lived microstructure alpha and implementation friction:

> Predictive information exists at very short horizons, but its economic value declines rapidly as round-trip trading costs increase.

---

# 15. Backtest Validation

The project includes explicit sanity checks for the main timing and portfolio assumptions.

The checks verify that:

- March observations precede the July OOS period.
- Model fitting uses training data only.
- Alpha-score thresholds are determined from March only.
- Entry timestamps occur strictly after signal timestamps.
- Exit timestamps occur strictly after entry timestamps.
- Trades do not overlap within individual symbols.
- Portfolio returns remain finite.
- NAV remains finite and positive.

The final validation reports:

```text
All leakage/timing/NAV sanity checks passed.
```

---

# 16. Key Results

The main empirical results can be summarized as follows.

### Predictive modelling

```text
Random Forest classification AUC:       ~0.74
Random Forest July alpha OOS AUC:       0.6865
Logistic Regression July OOS AUC:       0.6624
```

### Alpha ranking

```text
5-event Q5–Q1 forward-return spread:    ~1.91 bps
RF 1-event Rank IC:                     0.0857
RF 3-event Rank IC:                     0.0204
RF 5-event Rank IC:                     0.0123
RF 10-event Rank IC:                   -0.0009
```

### OOS systematic trading

```text
OOS trades:                             2,322
Average gross return/trade:             1.561 bps
Average net return/trade:               1.061 bps
Net hit rate:                           52.20%
Net per-trade Sharpe:                   0.019
Net OOS return:                         +0.3689%
Maximum net drawdown:                   -0.72%
```

### Cost sensitivity

```text
0.0 bp cost  → +0.6704%
0.5 bp cost  → +0.3689%
1.0 bp cost  → +0.0683%
2.0 bp cost  → -0.5304%
```

---

# 17. Research Interpretation

The project shows that engineered trade and order-book variables contain measurable information about extremely short-horizon price movements.

Three findings are particularly important.

### 1. Microstructure variables contain nonlinear predictive information

Random Forest outperforms the linear Logistic Regression benchmark, while feature importance shows that **trade pressure, directional volume and short-term momentum** account for most of the model's predictive structure.

### 2. Classification probabilities can function as alpha scores

The Random-Forest probability is not only useful for assigning a binary class. Sorting OOS observations by model score produces approximately **1.91 bps of spread between the highest and lowest signal quintiles** over the subsequent five events.

### 3. The alpha is highly transient

Random-Forest Rank IC reaches **0.0857 at the next event**, falls sharply after three events and is approximately zero by ten events.

The economic value of the signal therefore depends strongly on both execution horizon and transaction costs.

---

# 18. Technology Stack

The project is implemented in Python using:

```text
Python
├── pandas
├── NumPy
├── scikit-learn
│   ├── LogisticRegression
│   ├── RandomForestClassifier
│   ├── StandardScaler
│   ├── TimeSeriesSplit
│   └── ROC / classification metrics
├── TensorFlow / Keras
│   └── Feedforward Neural Network
├── SciPy
│   └── Spearman Rank Correlation
└── Matplotlib
    └── ROC, alpha-decay, NAV and drawdown visualisation
```

---

# 19. Project Workflow

```text
1. Load raw TAQ trade and quote files
2. Construct millisecond timestamps
3. Align trades with latest observable quotes
4. Restrict observations to opening-hour trading
5. Build market-microstructure features
6. Construct next-event directional labels
7. Train Logistic Regression, Random Forest and Neural Network classifiers
8. Evaluate ROC-AUC, accuracy, confusion matrices and threshold sensitivity
9. Rebuild chronological March / July alpha-research samples
10. Fit Logistic Regression and Random Forest on March only
11. Generate July OOS probability-based alpha scores
12. Sort signals into quintiles and measure forward returns
13. Compute Spearman Rank IC across event lags
14. Estimate alpha decay and signal persistence
15. Convert extreme RF scores into systematic long / short signals
16. Execute positions from the next market event
17. Apply fixed holding periods and non-overlapping trade rules
18. Include round-trip implementation costs
19. Aggregate trades into minute-level portfolio returns
20. Evaluate NAV, Sharpe, hit rate, drawdown and cost sensitivity
```

---

## Scope

This repository is designed as a **market-microstructure machine-learning and systematic alpha research workflow**.

It connects statistical prediction with downstream alpha validation:

```text
Prediction
    →
Ranking
    →
Persistence
    →
Trading Rule
    →
Economic Evaluation
```

The backtest is intentionally a compact signal-research framework rather than a full exchange-level HFT execution simulator.

---

## Reference

The original prediction exercise is motivated by research on machine learning in market microstructure and high-frequency trading, including:

> Michael Kearns & Yuriy Nevmyvaka — *Machine Learning for Market Microstructure and High Frequency Trading*