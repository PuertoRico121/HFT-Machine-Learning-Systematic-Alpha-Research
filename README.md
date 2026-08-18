# HFT Machine Learning & Systematic Alpha Research

## Overview

This project investigates whether **high-frequency market microstructure data** contain short-lived predictive information about equity price movements, and whether machine-learning probabilities can be converted into systematic alpha signals.

The workflow combines:

- Millisecond-level Trade and Quote (TAQ) data
- Trade–quote timestamp alignment
- Limit-order-book feature engineering
- Logistic Regression, Random Forest and feedforward Neural Networks
- Out-of-sample classification
- Probability-based alpha scoring
- Signal ranking and Rank IC analysis
- Alpha persistence / signal decay
- Systematic long–short signal construction
- Transaction-cost-aware backtesting

The research is structured around three questions:

1. Can microstructure variables predict the direction of the next mid-price movement?
2. Do model probabilities contain useful information for ranking future returns?
3. Does this predictive signal remain economically meaningful after execution delay and transaction costs?

---

## Research Pipeline

```text
Raw TAQ Trades & Quotes
        |
        v
Timestamp Alignment
        |
        v
Microstructure Feature Engineering
        |
        v
ML Directional Classification
        |
        v
Probability-Based Alpha Scores
        |
        v
Quintile & Rank IC Analysis
        |
        v
Signal Persistence / Alpha Decay
        |
        v
Systematic Long / Short Rules
        |
        v
Strict OOS Backtest
        |
        v
Sharpe, NAV, Drawdown & Cost Analysis
```

---

## Research Problem

At each trade event $t$, the model observes the latest available market state and predicts whether the next mid-price will be higher than the current mid-price.

The binary target is:

$$
y_t =
\begin{cases}
1, & \text{if } Mid_{t+1} > Mid_t \\
0, & \text{otherwise}
\end{cases}
$$

where the mid-price is:

$$
Mid_t = \frac{Ask_t + Bid_t}{2}
$$

Instead of using the model only as a hard classifier, the predicted probability

$$
P(y_t = 1 \mid X_t)
$$

is retained as a **continuous alpha score**.

This allows the project to study both classification performance and the economic information contained in model rankings.

---

## Data

The project uses millisecond-level **Trade and Quote (TAQ)** data.

Each trade observation contains information such as:

- Symbol
- Timestamp
- Transaction price
- Transaction size

Quote observations contain:

- Best bid
- Best ask
- Bid size
- Ask size
- Timestamp
- Symbol

The main analysis focuses on the first hour after the US market open:

```text
09:30 – 10:30
```

For the systematic alpha extension:

| Dataset | Date | Observations |
|---|---|---:|
| Training | 12 March 2024 | 51,053 |
| Strict OOS | 11 July 2024 | 44,782 |
| Universe | — | 11 stocks |

March is used for model estimation and signal calibration.

July remains fully out of sample.

---

## Trade–Quote Alignment

Trades and quotes arrive as separate event streams.

To reconstruct the market state visible when each trade occurred, trades are matched to the **latest available quote at or before the trade timestamp** using backward as-of alignment.

```python
pd.merge_asof(
    trades,
    quotes,
    on="TIMESTAMP",
    by="SYM_ROOT",
    direction="backward"
)
```

The resulting dataset represents:

```text
Trade Event
+ Latest Observable Quote
+ Bid / Ask Depth
+ Engineered Market-State Features
```

This creates the event-level feature matrix used by the machine-learning models.

---

# Microstructure Features

## Normalized Spread

The quoted spread is normalized by the current midpoint:

$$
Spread_t =
\frac{Ask_t - Bid_t}{Mid_t}
$$

This captures the current width of the market relative to the stock price.

---

## Volume Imbalance

Order-book imbalance measures relative depth on the bid and ask sides:

$$
Imbalance_t =
\frac{BidSize_t - AskSize_t}
{BidSize_t + AskSize_t}
$$

Positive values indicate greater displayed bid-side liquidity.

Negative values indicate greater ask-side liquidity.

---

## Microprice

A depth-weighted estimate of the current price is calculated as:

$$
Microprice_t =
\frac{
Ask_t \times BidSize_t
+
Bid_t \times AskSize_t
}{
BidSize_t + AskSize_t
}
$$

The alpha-research extension converts this into a displacement from the midpoint:

$$
MicropriceDisp_t =
\frac{Microprice_t - Mid_t}{Mid_t}
$$

This captures the direction in which order-book depth shifts the implied market price.

---

## Trade Pressure

The transaction price is compared with the prevailing midpoint and quoted spread.

A standardized trade-location measure is constructed as:

$$
TradeLocation_t =
\frac{
TradePrice_t - Mid_t
}{
(Ask_t-Bid_t)/2
}
$$

This captures whether trading activity is concentrated toward the bid or ask side.

---

## Signed Volume

Directional trade flow is represented using transaction direction and trade size.

In the alpha extension:

$$
SignedLogVolume_t =
TradeSign_t \times \log(1 + Size_t)
$$

The logarithmic transformation reduces the influence of unusually large trades.

---

## Short-Term Momentum

Recent event-level price movement is captured using:

$$
Momentum_t =
\frac{Mid_t}{Mid_{t-1}} - 1
$$

This represents the most recent direction of short-horizon price formation.

---

# Machine Learning Models

Three supervised-learning models are used in the original classification study.

## Logistic Regression

Logistic Regression provides an interpretable linear benchmark.

The workflow includes:

- Feature standardization
- L2 regularization
- Time-aware train/test splitting
- Coefficient analysis

Average performance:

| Metric | Result |
|---|---:|
| Train Accuracy | 68.59% |
| Test Accuracy | 71.91% |

Trade pressure has the largest estimated coefficient among the engineered features.

---

## Random Forest

Random Forest is used to capture nonlinear interactions between order-flow and market-state variables.

Main configuration:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_leaf=100,
    bootstrap=True
)
```

Performance:

| Metric | Result |
|---|---:|
| Train Accuracy | 71.53% |
| Test Accuracy | 74.04% |
| Train ROC-AUC | ~0.73 |
| Test ROC-AUC | ~0.74 |

### Feature Importance

| Feature | Importance |
|---|---:|
| Trade Pressure | ~38.5% |
| Signed Volume | ~24.7% |
| Momentum | ~22.8% |
| Spread | ~6.9% |
| Microprice / Smart Price | ~5.6% |
| Volume Imbalance | ~1.5% |

Trade pressure, signed volume and momentum account for roughly **86% of Random Forest feature importance**.

---

## Feedforward Neural Network

The neural network is a fully connected classifier rather than a recurrent sequence model.

Architecture:

```text
Input
  |
Dense(25) + ReLU
  |
Dense(10) + ReLU
  |
Dense(5) + ReLU
  |
Sigmoid Probability Output
```

Training uses:

- Adam optimizer
- Binary cross-entropy
- L2 regularization
- Mini-batch gradient descent
- 15 training epochs

The model achieves approximately:

| Metric | Result |
|---|---:|
| Test Accuracy | ~72.6% |
| Test ROC-AUC | ~0.71 |
| Cross-month ROC-AUC | ~0.70 |

---

# Class Imbalance & Threshold Analysis

Next-event upward price movements are less frequent than the non-upward class.

As a result, a default probability threshold of 0.5 can produce relatively high overall accuracy while missing many positive events.

The project therefore evaluates:

- Precision
- Recall
- F1-score
- Confusion matrices
- ROC-AUC
- Threshold sensitivity

For the Random Forest on the later-period evaluation:

### Threshold = 0.50

```text
Class-1 Recall: 13.61%
F1 Score:       0.220
Accuracy:       72.85%
```

### Threshold = 0.40

```text
Class-1 Recall: 68.44%
F1 Score:       0.516
Accuracy:       63.86%
```

The analysis therefore shifts attention away from hard classification and toward the information contained in the underlying probability score.

---

# Systematic Alpha Research Extension

The second stage of the project transforms the machine-learning models into **alpha scorers**.

The research design uses a strict chronological split:

```text
March 2024
    |
    |-- Fit preprocessing
    |-- Fit models
    |-- Estimate score distribution
    |-- Define trading thresholds
    |
    v
July 2024
    |
    |-- Frozen preprocessing
    |-- Frozen models
    |-- Frozen thresholds
    |
    v
Strict Out-of-Sample Evaluation
```

A liquidity filter is also applied:

$$
NormalizedSpread \leq 0.002
$$

---

## Alpha Model Performance

The alpha extension focuses on Logistic Regression and Random Forest.

| Model | March Train AUC | July OOS AUC |
|---|---:|---:|
| Logistic Regression | 0.6734 | 0.6624 |
| Random Forest | 0.7113 | **0.6865** |

Random Forest provides the stronger nonlinear OOS ranking signal and is used as the primary alpha model.

---

# Probability-Based Alpha Score

For every July event, Random Forest produces:

$$
AlphaScore_t = P(y_t=1 \mid X_t)
$$

Rather than immediately transforming this probability into a long or short position, the observations are first ranked by score.

The OOS sample is divided into five signal quintiles.

### 5-Event Forward Returns

| Alpha Quintile | Mean Forward Return |
|---|---:|
| Q1 | -2.673 bps |
| Q2 | -2.337 bps |
| Q3 | -2.632 bps |
| Q4 | -1.713 bps |
| Q5 | -0.767 bps |

The extreme-quintile spread is:

$$
Q5-Q1
=
-0.767 - (-2.673)
=
1.906 \text{ bps}
$$

The model score therefore contains useful information for ranking subsequent short-horizon returns.

Although the intermediate quintiles are not perfectly monotonic, the highest-scoring events materially outperform the lowest-scoring events.

---

# Signal Persistence & Alpha Decay

The next question is how long this predictive information survives after the signal is generated.

Signal persistence is measured using **Spearman Rank Information Coefficient**.

For forecast lag $h$:

$$
IC_h =
Corr_{rank}
\left(
AlphaScore_t,
Return_{t+h \rightarrow t+h+1}
\right)
$$

The current model score is compared with single-event returns occurring after different forecast lags.

### OOS Rank IC

| Forecast Lag | Logistic Regression | Random Forest |
|---|---:|---:|
| 1 event | 0.0816 | **0.0857** |
| 3 events | 0.0218 | 0.0204 |
| 5 events | 0.0123 | 0.0123 |
| 10 events | 0.0011 | -0.0009 |

The signal decay is substantial:

```text
1 Event
RF Rank IC = 0.0857
        |
        v
3 Events
RF Rank IC = 0.0204
        |
        v
5 Events
RF Rank IC = 0.0123
        |
        v
10 Events
RF Rank IC ≈ 0
```

The model therefore captures **highly transient microstructure alpha**.

Most predictive information is concentrated immediately after signal formation and largely disappears within ten subsequent market events.

---

# Systematic Signal Construction

The final strategy trades only extreme Random-Forest scores.

Thresholds are estimated exclusively from the March training distribution.

```text
Bottom 5% → Short
Top 5%    → Long
Middle    → No Trade
```

March-derived thresholds:

| Signal | Probability Threshold |
|---|---:|
| Short | ≤ 0.0978 |
| Long | ≥ 0.5098 |

The same thresholds are frozen and applied to the July OOS sample.

---

# Execution Logic

For a signal generated at event $t$:

```text
Signal at t
    |
    v
Entry at t+1
    |
    v
Hold for 10 market events
    |
    v
Exit
```

The backtest additionally enforces:

- No same-event execution
- No overlapping positions within the same symbol
- Fixed holding horizon
- Frozen OOS signal thresholds
- Round-trip transaction costs

The baseline implementation-cost assumption is:

$$
Cost = 0.5 \text{ bp per round trip}
$$

---

# Out-of-Sample Trading Results

The July OOS sample generates:

| Position | Trades |
|---|---:|
| Long | 1,524 |
| Short | 798 |
| Total | **2,322** |

### Trade-Level Performance

| Metric | Result |
|---|---:|
| Average Gross Return | **1.561 bps / trade** |
| Average Net Return | **1.061 bps / trade** |
| Gross Hit Rate | 54.48% |
| Net Hit Rate | 52.20% |
| Gross Per-Trade Sharpe | 0.028 |
| Net Per-Trade Sharpe | 0.019 |

The trade-level Sharpe measure is calculated as:

$$
Sharpe_{trade}
=
\frac{
Mean(Return_{trade})
}{
Std(Return_{trade})
}
$$

It is therefore a per-trade mean-to-volatility ratio rather than an annualized portfolio Sharpe ratio.

---

# Portfolio Construction

Trades occurring within the same one-minute interval are first aggregated using equal weights.

```text
Individual Trades
      |
      v
1-Minute Equal-Weighted Return
      |
      v
Chronological Compounding
      |
      v
Portfolio NAV
```

### July OOS Performance

| Metric | Result |
|---|---:|
| Gross Ending NAV | 1.006704 |
| Net Ending NAV | 1.003689 |
| Gross Return | **+0.6704%** |
| Net Return | **+0.3689%** |
| Maximum Net Drawdown | **-0.72%** |

---

# Transaction Cost Sensitivity

Short-horizon alpha is highly sensitive to implementation costs.

The same frozen OOS trades are therefore evaluated under different round-trip cost assumptions.

| Round-Trip Cost | OOS Return |
|---:|---:|
| 0.0 bp | **+0.6704%** |
| 0.5 bp | **+0.3689%** |
| 1.0 bp | **+0.0683%** |
| 2.0 bp | **-0.5304%** |

The signal remains marginally positive at 1 bp of implementation cost but becomes negative at 2 bps.

This is consistent with the rapid Rank IC decay observed in the signal-persistence analysis.

---

# Validation Checks

The research pipeline includes explicit checks for chronological integrity and trading logic.

The notebook verifies that:

- March training observations occur before July OOS observations
- Model estimation uses training data only
- Trading thresholds are estimated from March only
- Entry occurs after signal generation
- Exit occurs after entry
- Trades do not overlap within the same symbol
- Portfolio returns are finite
- NAV remains finite and positive

```text
All leakage / timing / NAV sanity checks passed.
```

---

# Key Results

## Machine Learning

```text
Random Forest classification ROC-AUC:   ~0.74
Random Forest strict OOS ROC-AUC:        0.6865
Logistic Regression strict OOS ROC-AUC: 0.6624
```

## Alpha Research

```text
5-event Q5–Q1 return spread:             1.906 bps
RF 1-event Rank IC:                      0.0857
RF 3-event Rank IC:                      0.0204
RF 5-event Rank IC:                      0.0123
RF 10-event Rank IC:                    -0.0009
```

## Systematic Strategy

```text
OOS trades:                              2,322
Average gross return / trade:            1.561 bps
Average net return / trade:              1.061 bps
Net hit rate:                            52.20%
Net per-trade Sharpe:                    0.019
Net OOS return:                          +0.3689%
Maximum drawdown:                        -0.72%
```

---

# Main Findings

### 1. Market microstructure features contain short-horizon predictive information

Trade pressure, directional volume and local momentum are the strongest drivers of model predictions.

Random Forest improves on the linear Logistic Regression benchmark, suggesting that nonlinear relationships between microstructure variables contain additional information.

### 2. Prediction probabilities contain alpha-ranking information

The Random-Forest probability can be used as a continuous alpha score.

The highest and lowest OOS signal quintiles differ by approximately **1.91 bps** in subsequent five-event returns.

### 3. Alpha persistence is extremely short

Random-Forest Rank IC is **0.0857 at the next market event**, but falls rapidly and approaches zero after ten events.

The signal therefore behaves as short-lived microstructure alpha rather than a persistent medium-horizon factor.

### 4. Trading economics depend strongly on implementation costs

The strategy remains positive under low transaction-cost assumptions, but the return is almost eliminated at 1 bp and becomes negative at 2 bps.

The results highlight the connection between:

```text
Alpha Strength
      +
Signal Decay
      +
Execution Horizon
      +
Transaction Costs
      =
Realized Trading Economics
```

---

# Technology Stack

```text
Python
│
├── pandas
│   └── TAQ processing and timestamp alignment
│
├── NumPy
│   └── Numerical feature construction
│
├── scikit-learn
│   ├── Logistic Regression
│   ├── Random Forest
│   ├── StandardScaler
│   ├── TimeSeriesSplit
│   └── Classification / ROC metrics
│
├── TensorFlow / Keras
│   └── Feedforward Neural Network
│
├── SciPy
│   └── Spearman Rank IC
│
└── Matplotlib
    └── ROC, signal-decay, NAV and drawdown analysis
```

---

# Research Scope

This repository is a **market-microstructure machine-learning and systematic alpha research project**.

Its main objective is to connect:

```text
Feature Engineering
       ↓
Predictive Modelling
       ↓
Alpha Scoring
       ↓
Signal Persistence
       ↓
Systematic Trading Rules
       ↓
Economic Validation
```

The backtest is designed as a compact research framework for testing predictive signals and trading economics rather than as a production exchange-level HFT execution simulator.

---

## Reference

The original classification research is motivated by work on machine learning and high-frequency market microstructure, including:

**Michael Kearns & Yuriy Nevmyvaka — Machine Learning for Market Microstructure and High Frequency Trading**