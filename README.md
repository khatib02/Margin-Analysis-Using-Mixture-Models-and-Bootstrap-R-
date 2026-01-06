# Statistical Analysis of Retail Margins: A Mixture Model Approach

![R](https://img.shields.io/badge/Language-R-blue)
![Status](https://img.shields.io/badge/Status-Complete-green)
![Focus](https://img.shields.io/badge/Focus-Statistical%20Inference-orange)

## Executive Summary
This project presents an end-to-end statistical analysis of a retail transaction dataset. Unlike standard machine learning approaches that prioritize "black-box" predictive accuracy, this analysis focuses on **structural interpretability**, **robustness**, and **statistical soundness**.

**Key Outcome:**
The final model identifies **two distinct latent economic regimes (clusters)** within the transactions, allowing the business to separate high-risk/low-margin orders from stable high-performance transactions. This was achieved using a **Finite Mixture of Gaussian Linear Regressions**, selected after rigorously comparing Generalized Additive Models (GAMLSS) and Gradient Boosting methods.

---

## Business Problem
Retail transaction data is characterized by high complexity: time effects, geographic heterogeneity, and highly skewed financial variables. The objective was to determine which factors drive **Profit Margins** defined as:

$$\text{Margin} = \frac{\text{Profit}}{\text{Sales}}$$

**Goals:**
* Construct an analytically coherent dataset.
* Balance predictive stability with business interpretability.
* Provide actionable insights on profitability drivers.

---

## Methodology & Technical Approach

The workflow was implemented in **R**, combining classical statistical reasoning with modern machine learning tools.

### 1. Data Engineering & Rigorous Cleaning
* **Decomposition:** Broken down complex identifiers (Order ID, Product ID) to remove redundancy using string functions.
* **Dimensionality Reduction:** Aggregated rare geographic categories into census-level divisions to reduce noise.
* **Data type modifications:** Variables were explicitly cast to appropriate data types based on their semantic meaning and analytical role.

### 2. Model Selection Journey
I evaluated a sequence of increasingly flexible models to understand the distributional structure of the data:

| Model Type | Purpose | Finding |
| :--- | :--- | :--- |
| **Baseline Linear (Lasso/Ridge)** | Reference point | Residuals showed heavy tails; Gaussian assumption failed. |
| **GAMLSS (Student-t)** | Handle heavy tails | Improved fit, but residuals showed bimodality. |
| **Tree-Based (XGBoost/LightGBM)** | Maximize prediction | High accuracy, but low interpretability for specific transaction drivers. |
| **Finite Mixture Model (Final)** | **Capture heterogeneity** | Successfully modeled two latent subpopulations with distinct risk profiles. |

---

## Key Findings: The "Two-Cluster" Discovery

The Finite Mixture Model revealed that transactions naturally fall into two clusters, which provides a direct link to business risk assessment:

* **Cluster 1 (High Risk):** Characterized by lower average margins and a high probability of negative outcomes (losses).
* **Cluster 2 (High Performance):** Exhibits substantially higher expected margins and lower downside risk.

<img width="1723" height="1069" alt="image" src="https://github.com/user-attachments/assets/ae155bbb-7521-4417-9c84-5075740c9566" />

> *Figure: Risk-Return trade-off between the two identified clusters.*

### Drivers of Profitability
Using a concomitant model, we identified the structural drivers that increase the odds of a transaction belonging to the "High Performance" cluster. This allows for strategic decision-making, such as adjusting discount strategies for specific order profiles.

---

## Advanced Statistical Validation

To ensure the model wasn't just "fitting noise," I performed rigorous stability checks rarely seen in standard analyses:

1.  **Convergence Analysis:** Tested the EM algorithm stability across varying numbers of random initializations ($n_{rep}$). Results showed the solution stabilizes at $n_{rep} \approx 20$.
2.  **Nonparametric Bootstrap ($B=1000$):**
    * Implemented parallelized bootstrap refitting to assess generalization.
    * Constructed empirical confidence intervals for cluster-specific expected margins.
    * Confirmed that the clusters correspond to genuinely different distributions, not statistical artifacts.

---

## Tech Stack
* **Language:** R
* **Data Manipulation:** `dplyr`, `tidyr`, `purrr`.
* **Visualization:** `ggplot2`.
* **Modeling:** `flexmix` (Mixture Models), `gamlss`, `xgboost`, `lightgbm`, `glmnet`.
* **Parallel Computing:** `foreach`, `doParallel`.

---

## 🚀 Conclusion
The final Gaussian Mixture Model represents a robust compromise between statistical rigor and business relevance. It moves beyond simple point predictions to provide a probabilistic understanding of operational risk, enabling data-driven policy adjustments.

<img width="596" height="188" alt="image" src="https://github.com/user-attachments/assets/b56a8bf8-65a1-4b43-ae71-9bfe736fecd1" />

