# Statistical Analysis of Retail Margins: A Mixture Model Approach

![R](https://img.shields.io/badge/Language-R-blue)
![Status](https://img.shields.io/badge/Status-Complete-green)
![Focus](https://img.shields.io/badge/Focus-Statistical%20Inference-orange)

## Executive Summary
I analyzed a superstore transaction dataset to understand profit margins. I avoided "black-box" machine learning methods that only focus on prediction. Instead, I built a model that emphasizes **structure** and **reliability**.

**Key Outcome:**
The final model identifies **two distinct economic groups (clusters)** within the data. This helps the business separate high-risk orders from stable, high-performance ones. I achieved this using a **Finite Mixture of Gaussian Linear Regressions**, chosen after comparing it with Generalized Additive Models (GAMLSS) and Gradient Boosting.

---

## Business Problem
Retail data is complex. It involves time effects, location differences, and skewed financial numbers. My goal was to find the factors that drive **Profit Margins**:

$$\text{Margin} = \frac{\text{Profit}}{\text{Sales}}$$

**Goals:**
* Build a clean, consistent dataset.
* Balance stable predictions with business meaning.
* Find practical drivers of profit.

---

## Methodology & Technical Approach

I used **R** for the entire workflow. I combined standard statistical reasoning with modern tools.

### 1. Data Engineering & Cleaning
* **Decomposition:** I broke down complex Order and Product IDs to remove repetition.
* **Dimensionality Reduction:** I grouped rare locations into larger census divisions to reduce noise.
* **Data Types:** I set specific data types for all variables based on their analytical role.

### 2. Model Selection Journey
I tested a sequence of models to understand the data structure:

| Model Type | Purpose | Finding |
| :--- | :--- | :--- |
| **Baseline Linear (Lasso/Ridge)** | Reference point | Residuals had heavy tails. The Gaussian assumption failed. |
| **GAMLSS (Student-t)** | Handle heavy tails | Fit improved, but residuals showed two peaks (bimodality). |
| **Tree-Based (XGBoost/LightGBM)** | Maximize prediction | Accuracy was high, but the drivers were hard to explain. |
| **Finite Mixture Model (Final)** | **Capture heterogeneity** | Modeled the two latent groups with distinct risk profiles. Model with the highest accuracy and intepretability |

---

## Key Findings: The "Two-Cluster" Discovery

The Finite Mixture Model shows that transactions fall into two groups. This links directly to business risk:

* **Cluster 1 (High Risk):** Has lower average margins and a high chance of loss.
* **Cluster 2 (High Performance):** Shows higher expected margins and low risk.

<img width="1530" height="1350" alt="image" src="https://github.com/user-attachments/assets/a5e49e1f-9626-4335-a656-da1c601d3495" />

> *Figure: Risk-Return trade-off. Cluster 2 (right) shows higher expected margin and lower probability of negative margin compared to Cluster 1 (left).*

### Drivers of Profitability
I used a concomitant model to find what puts a transaction in the "High Performance" cluster. This helps with decisions like setting discount rates or changing shipping strategies.

<img width="1530" height="1350" alt="image" src="https://github.com/user-attachments/assets/d2cfe6df-1e74-463e-9f5a-f202b0a6645b" />

> *Figure: Operational drivers (e.g., Discount, Sub-Category, Shipping Mode) that most strongly influence the probability of high-performance classification.*

---

## Advanced Statistical Validation

I ran stability checks to prove the model works reliably.

### 1. Diagnostic Robustness
Comparing the residuals of the baseline linear model against the mixture model highlights the necessity of the chosen approach.

<img width="1530" height="1350" alt="image" src="https://github.com/user-attachments/assets/d7c83df4-82e9-4a70-8e0d-146c371fc4f6" />

> *Figure: The Linear Model (Left) fails to capture extreme tails. The Mixture Model (Right) adheres more closely to the theoretical distribution.*

### 2. Convergence & Bootstrap Analysis
* **Convergence Analysis:** I tested the EM algorithm stability. The solution becomes stable after about 20 random starts ($n_{rep} \approx 20$).
* **Nonparametric Bootstrap ($B=1000$):**
    * I re-ran the model 1000 times on resampled data.
    * I built confidence intervals for the expected margins of each cluster.
    * This confirmed the clusters are real and not just statistical noise.

<img width="1530" height="1350" alt="image" src="https://github.com/user-attachments/assets/9dfbd59b-5bec-4e46-861d-c75ad4a9ab53" />

> *Figure: Bootstrap distributions of RMSE and predictive $R^2$ showing stable out-of-sample performance.*

---

## Tech Stack
* **Language:** R
* **Data Manipulation:** `tidyr`
* **Visualization:** `ggplot2`
* **Modeling:** `flexmix` (Mixture Models), `gamlss`, `xgboost`, `lightgbm`, `glmnet`
* **Parallel Computing:** `foreach`, `doParallel`

---

## Conclusion
The final Gaussian Mixture Model balances statistical rigor with business needs. It provides a clear view of operational risk and supports better decision-making.

<img width="1194" height="377" alt="image" src="https://github.com/user-attachments/assets/8040159c-84da-40f7-9e23-a854cd3a58be" />


> *Out-of-sample performance metrics sorted by descending predictive $R^2$. While boosting methods (XGBoost/LightGBM) achieved high raw accuracy, the Gaussian Mixture Model achieved the highest accuracy and interpretability of the bussiness model*





