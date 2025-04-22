# Renewable Energy Forecasting Using Machine Learning

## Project Summary

This project uses supervised machine learning techniques to predict the annual share of renewable energy adoption across countries using historical data from Kaggle from 2010 to 2023. The modeling approach incorporates linear, regularized, and ensemble methods to evaluate the predictive power of infrastructure investment, energy production, environmental impact, and job creation metrics.

The analysis is designed to assess data readiness, feature relevance, and model performance in capturing complex drivers of renewable energy transitions.

---

## Objectives

- Forecast US renewable energy share using multivariate indicators.
- Evaluate and compare the performance of regression-based ML models.
- Apply feature engineering to enhance interpretability and model accuracy.
- Visualize correlations, outliers, and model diagnostics to support analysis.

---

## Data Overview

**Source:** [Kaggle – Renewable Energy Adoption & Climate Change Response (2010–2023)](https://www.kaggle.com/datasets/ravixedmlover/renewable-energy-adoption-and-climate-change-resp)

**Key Features:**
- `Energy Production (MWh)`
- `Installed Capacity (MW)`
- `Investment in Renewable Infrastructure (USD)`
- `CO₂ Reduction (tons)`
- `Jobs Created`
- `Country/Region`, `Energy Source`, `Year`
- `Renewable Energy Share (%)`

**Engineered Variables:**
- Investment per MW installed  
- Investment per ton of CO₂ reduced  
- Capacity factor  
- Lag features: prior year investment & renewable share  
- Log and squared transformations for non-linearity

---

## Methodology

### 1. Preprocessing Pipeline
Custom utility functions were implemented for:
- Data validation and summary
- Min-max normalization, standardization
- One-hot encoding of categorical variables

### 2. Model Development
Models were trained across 5 train/test splits (10%, 25%, 50%, 75%, 90%) to simulate performance under varying data availability:

**Baseline Models:**
- Linear Regression  
- Ridge Regression (via `GridSearchCV`)  
- Lasso Regression (high max iteration)  

**Advanced Models:**
- Random Forest Regressor  
- XGBoost Regressor  

### 3. Evaluation Metrics
All models were evaluated using:
- RMSE (Root Mean Squared Error)  
- R² (Coefficient of Determination)  
- MAE (Mean Absolute Error)  
- MAPE (Mean Absolute Percentage Error)  

### 4. Visual Diagnostics
- Correlation heatmaps  
- R² matrix visualizations  
- Actual vs. predicted scatter plots  
- Residual plots  

---

## Results Summary

Across all model types and splits, predictive performance was limited due to weak signal strength in the features. Random Forest achieved the highest R² score of 0.04 with 90% training data.

| Training % | Linear | Ridge | Lasso | Random Forest | XGBoost |
|------------|--------|-------|-------|----------------|---------|
| 10%        | -1.17  | 0.00  | 0.00  | -0.04          | -0.10   |
| 50%        | -0.14  | 0.00  | 0.00  | -0.02          | -0.02   |
| 90%        | -0.10  | 0.01  | -0.02 | **0.04**       | -0.02   |

---

## Limitations & Future Work

- Current features omit geopolitical, policy, and grid-integration complexities.
- R² scores suggest weak predictive signals using static features alone.
- Future models should incorporate time-series dependencies, macroeconomic indicators, and spatial data granularity for improved accuracy.

---

## Contact
Edgar Venegas
University of Michigan – School for Environment and Sustainability
Specialization: Sustainable Systems
Email: evenegas@umich.edu
