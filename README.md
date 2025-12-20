![R](https://img.shields.io/badge/R-4.3.1-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

# Superstore Profit Margin Analysis

## Overview
This project performs **data analysis and modeling of profit margins** using Kaggle's Superstore dataset. The goal is to explore key factors affecting profitability and build predictive models to estimate margins for different product categories and customer segments.

## Objectives
- Perform **exploratory data analysis (EDA)** to understand sales, profit, and discount patterns.
- Identify **key drivers of profit margins** using statistical analysis.
- Build **predictive models** to estimate profit margins for different sales scenarios.
- Provide **actionable insights** for business decisions.

## Tools & Technologies
- **R**: For EDA and statistical modeling

## Dataset
The dataset is sourced from [Kaggle: Superstore Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final).  
It contains sales data including sales amount, category of products, customer, discounts, profit margins and customers location.

## Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/superstore-profit-analysis.git


## Key Insights
- The Gaussian mixture model achieved the best predictive performance.
- Bootstrap was used to allow inference without assuming normality of residuals.
- The model revealed two distinct distributions: one corresponding to a negative expected margin and the other to a positive expected margin.
- The cluster associated with a negative margin is primarily linked to discounts of 30%–40% and 50%–70%, as well as products such as furniture, bookcases, machines, and supplies.
- The cluster associated with a positive margin is mainly related to office supplies, art, and phones.


