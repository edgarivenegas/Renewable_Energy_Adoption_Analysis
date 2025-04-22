#%% [markdown]
## Predicting Renewable Energy Growth Using Machine Learning and Policy Analysis
# ---
#
# **Author:** Edgar Venegas 
#
# **Estimated Duration:** 40 hours 
#
# **Elapsed Time:** 45 hours  
# 
# ---
# #### **Objective:** 
# The global transition to renewable energy presents a complex set of challenges and opportunities. This project explores the use of machine learning models to predict the share of renewable energy across countries based on infrastructure investments, capacity, production, and environmental metrics. Using a data set from 2010 to 2023, the study applies Linear, Ridge, Lasso, Random Forest, and XGBoost models to evaluate the performance and abilities of renewable forecasting efforts.# 
#
#  ---
# #### **Tools & Libraries**  
# - **Data Handling:** pandas, numpy  
# - **Visualization:** matplotlib, seaborn  
# - **Machine Learning:** scikit-learn, Random Forest, XGBoost  
# ---
# #### **Data Source**  
# Kaggle -> *Renewable Energy Adoption & Climate Change Response (2010-2023)*  
# [Dataset Link](https://www.kaggle.com/datasets/ravixedmlover/renewable-energy-adoption-and-climate-change-resp)
#
#---------------------------------------------------------------------------------------
# SECTION 1 – Load & Summarize Dataset
#       
#---------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from utils import plot_linechart, plot_correlations, plot_rsquared, plot_act_v_pred, plot_residuals
from utils import summarize_data, preprocess_pipeline, train_test_splits, linear_reg_pred
from utils import eval_model


# Set up project folders
base_dir = Path.cwd()
out_dir = base_dir / 'output'
MODEL_VIS = out_dir / 'model_visuals'
EDA_DIR = out_dir / 'EDA'

for path in [MODEL_VIS, EDA_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Load, Check, and Summarize Dataset
df = pd.read_csv('renewable_energy_adoption.csv')
assert type(summarize_data) != None

df.head()
#%%
# Cleaning and Describing Features 
df['Environmental Impact Numeric (ton CO2 Reduction)'] = df['Environmental Impact (CO2 Reduction)'].str.extract(r'(\d+)').astype(int)
df['Year'] = df['Year'].astype(str)
'''
All countries have policy changes for the listed renewable energy sources per year!
'''
df = df.drop('Environmental Impact (CO2 Reduction)',axis=1)
df = df.drop('Policy Changes',axis=1)

summarize_data(df)
print('\nDatset Description:')
df.describe()
#%%[markdown]
#---------------------------------------------------------------------------------------
# SECTION 2 – Feature Engineering, Scaling, and Visualization Parameters
#       
#---------------------------------------------------------------------------------------
#%%
# Investment in Renewable Infrastructure per MW of Installed Capacity ($/MW)
df['Investment per MW of Installed Capacity ($/MW)'] = df['Investment in Renewable Infrastructure (USD)'] / df['Installed Capacity (MW)']

# Investment in Renewable Infrastructure per ton of CO2 Reduced ($/ton CO2 Reduction)
df['Investment per ton CO2 Reduced ($/ton CO2)']  = df['Investment in Renewable Infrastructure (USD)']/df['Environmental Impact Numeric (ton CO2 Reduction)']

# Capacity Factor
df['Capacity Factor'] = (df['Energy Production (MWh)']/df['Installed Capacity (MW)'])/8760

# Jobs Created per Dollar Invested in Renewable Infrastructure
df['Jobs per Investment in Renewables (Jobs/$)'] = df['Investment in Renewable Infrastructure (USD)'] / df['Jobs Created']

# Historical influence terms 
df['Prior Year Renewable Share (%)'] = df['Renewable Energy Share (%)'].shift(1)
df['Prior Year Investment'] = df['Investment in Renewable Infrastructure (USD)'].shift(1)

# Cleaning NaNs
df.dropna(subset=['Prior Year Renewable Share (%)', 'Prior Year Investment'], inplace=True)

# Square terms for positive or negative accelerating effects
df['Investment 2'] = df['Investment in Renewable Infrastructure (USD)'] ** 2
df['Capacity 2'] = df['Installed Capacity (MW)'] ** 2
df['Energy Production 2'] = df['Energy Production (MWh)'] ** 2

# Log transforms for normalization or multiplicative growth modeling
df['log_Investment'] = np.log1p(df['Investment in Renewable Infrastructure (USD)'])
df['log_CO2_Reduction'] = np.log1p(df['Environmental Impact Numeric (ton CO2 Reduction)'])


# Parameters for visualization
lc_params = [
    {'save_filename': 'energy_production_by_country.png',
        'chart_title': 'Renewable Energy Production',
        'y_name': 'Energy Production (MWh)',
        'fig_num': 1
    },
    {'save_filename': 'investment_renewable_infrastructure_by_country.png',
        'chart_title': 'Investment in Renewable Infrastructure',
        'y_name': 'Investment in Renewable Infrastructure (USD)',
        'fig_num': 2
    },
    {'save_filename': 'environmental_impact.png',
        'chart_title': 'Environmental Impact (CO2 Reduction)',
        'y_name': 'Environmental Impact Numeric (ton CO2 Reduction)',
        'fig_num': 3
    },
    {'save_filename': 'installed_capcity.png',
        'chart_title': 'Installed Capacity',
        'y_name': 'Installed Capacity (MW)',  
        'fig_num': 4
    },
    {'save_filename': 'renewables_share.png',
        'chart_title': 'Renewable Share',
        'y_name': 'Renewable Energy Share (%)', 
        'fig_num': 5
    },
]

#%%

# EDA Visualizations
# **Line Charts**
for params in lc_params:
    figsize = (12,6)
    plot_linechart(
        df, 
        'Year', 
        params['y_name'], 
        'Country/Region', 
        figsize, 
        params['chart_title'], 
        params['save_filename'],
        EDA_DIR, 
        params['fig_num']
)
    

#%%
# **Heatmaps**
plot_correlations(
    data=df,
    chart_title='Coefficient of Correlation Heatmap',
    cmap='coolwarm',
    figsize=(24,22),
    save_filename='coef_corr_heatmap.png',
    output_filepath=EDA_DIR,
    annotate=True
)

plot_rsquared(
    data=df,
    chart_title='Coefficient of Determination Heatmap',
    cmap='coolwarm',
    figsize=(24,22),
    save_filename='rsquared_heatmap.png',
    output_filepath=EDA_DIR,
    annotate=True
)
#%%[markdown]
#---------------------------------------------------------------------------------------
# SECTION 3 – Normalization, and Standardization
#       
#---------------------------------------------------------------------------------------
#%%
# Categorizing Features and onehot encoding
cat_features = []
quant_features = []

for col in df.columns:
    if df[col].dtype == 'object':
        cat_features.append(col)
    else:
        quant_features.append(col)

# Remove target variable from feature matrix
quant_features.remove('Renewable Energy Share (%)')

# Normalization, Standardization, and full preprocessing pipeline
X_processed,y = preprocess_pipeline(
    data=df,
    category_features=cat_features,
    quantitative_features=quant_features,
    target_label='Renewable Energy Share (%)',
    show_features=False)


#%%
# Test-Train datasubet spits
train_sizes = [0.1,0.25,0.50,0.75,0.9]
train_test_dataset = train_test_splits(X_processed,y,train_sizes)

#%%[markdown]
#---------------------------------------------------------------------------------------
# SECTION 4 – Model Development and Linear Regressions
#       
#---------------------------------------------------------------------------------------
#%%
#-----------------------------------------------|
#      Baseline 1: Linear Regression Model      |
#-----------------------------------------------|
# Calculate trainig and testing subsets
#
# Training set 10% -- Testing Set 90%
X_train_10, X_test_10, y_train_10, y_test_10 = train_test_dataset[0.1][0] ,train_test_dataset[0.1][1],train_test_dataset[0.1][2], train_test_dataset[0.1][3]

# Training set 20% -- Testing Set 80%
X_train_25, X_test_25, y_train_25, y_test_25 = train_test_dataset[0.25][0] ,train_test_dataset[0.25][1],train_test_dataset[0.25][2], train_test_dataset[0.25][3]

# Training set 50% -- Testing Set 50%
X_train_50, X_test_50, y_train_50, y_test_50 = train_test_dataset[0.5][0] ,train_test_dataset[0.5][1],train_test_dataset[0.5][2], train_test_dataset[0.5][3]

# Training set 75% -- Testing Set 25%
X_train_75, X_test_75, y_train_75, y_test_75 = train_test_dataset[0.75][0] ,train_test_dataset[0.75][1],train_test_dataset[0.75][2], train_test_dataset[0.75][3]

# Training set 90% -- Testing Set 10%
X_train_90, X_test_90, y_train_90, y_test_90 = train_test_dataset[0.9][0] ,train_test_dataset[0.9][1],train_test_dataset[0.9][2], train_test_dataset[0.9][3]

#%%
lr_pred_90 = linear_reg_pred(X_train_10, X_test_10, y_train_10)
lr_pred_75 = linear_reg_pred(X_train_25, X_test_25, y_train_25)
lr_pred_50 = linear_reg_pred(X_train_50, X_test_50, y_train_50)
lr_pred_25 = linear_reg_pred(X_train_75, X_test_75, y_train_75)
lr_pred_10 = linear_reg_pred(X_train_90, X_test_90, y_train_90)

# Linear Regression Evaluation
lr_eval_10 = eval_model(y_test_10,lr_pred_90)
lr_eval_25 = eval_model(y_test_25,lr_pred_75)
lr_eval_50 = eval_model(y_test_50,lr_pred_50)
lr_eval_75 = eval_model(y_test_75,lr_pred_25)
lr_eval_90 = eval_model(y_test_90,lr_pred_10)

lr_rmse_10, lr_r2_10, lr_mae_10, lr_mape_10 = lr_eval_10['rmse'], lr_eval_10['r2'], lr_eval_10['mae'], lr_eval_10['mape']
lr_rmse_25, lr_r2_25, lr_mae_25, lr_mape_25 = lr_eval_25['rmse'], lr_eval_25['r2'], lr_eval_25['mae'], lr_eval_25['mape']
lr_rmse_50, lr_r2_50, lr_mae_50, lr_mape_50 = lr_eval_50['rmse'], lr_eval_50['r2'], lr_eval_50['mae'], lr_eval_50['mape']
lr_rmse_75, lr_r2_75, lr_mae_75, lr_mape_75 = lr_eval_75['rmse'], lr_eval_75['r2'], lr_eval_75['mae'], lr_eval_75['mape']
lr_rmse_90, lr_r2_90, lr_mae_90, lr_mape_90 = lr_eval_90['rmse'], lr_eval_90['r2'], lr_eval_90['mae'], lr_eval_90['mape']


print('The evaluation metrics for the linear regression with a: \n')
print(f'10% training set are:\nRMSE: {lr_rmse_10:.2f}\nR squared: {lr_r2_10:.2f}\nMAE: {lr_mae_10:2f}\nMAPE: {lr_mape_10*100:.2f}%\n')
print(f'25% training set are:\nRMSE: {lr_rmse_25:.2f}\nR squared: {lr_r2_25:.2f}\nMAE: {lr_mae_25:2f}\nMAPE: {lr_mape_25*100:.2f}%\n')
print(f'50% training set are:\nRMSE: {lr_rmse_50:.2f}\nR squared: {lr_r2_50:.2f}\nMAE: {lr_mae_50:2f}\nMAPE: {lr_mape_50*100:.2f}%\n')
print(f'75% training set are:\nRMSE: {lr_rmse_75:.2f}\nR squared: {lr_r2_75:.2f}\nMAE: {lr_mae_75:2f}\nMAPE: {lr_mape_75*100:.2f}%\n')
print(f'90% training set are:\nRMSE: {lr_rmse_90:.2f}\nR squared: {lr_r2_90:.2f}\nMAE: {lr_mae_90:2f}\nMAPE: {lr_mape_90*100:.2f}%')


#-----------------------------------------------|
#     Baseline 1: Linear Regression Visuals     |
#-----------------------------------------------|
#
# Plot 90% testing Set
plot_act_v_pred(
    y_test=y_test_90,
    prediction=lr_pred_10,
    x_label='Actual Renewable Energy Share (%)',
    y_label='Predicted Renewable Energy Share (%)',
    chart_title='Linear Regression Performance',
    xtick_size=12,
    ytick_size=12,
    chart_title_size=18,
    save_filename='linear_regression_act_pred',
    output_filepath=MODEL_VIS)
plot_residuals(
    y_test_90,
    lr_pred_10,
    x_label='Actual Renewable Energy Share (%)',
    y_label='Predicted Renewable Energy Share (%)',
    chart_title='Linear Regression Performance',
    xtick_size=12,
    ytick_size=12,
    chart_title_size=18,
    save_filename='linear_regression_residuals',
    output_filepath=MODEL_VIS)

# %%
#-----------------------------------------------|
#          Baseline 2: Ridge Regression         |
#-----------------------------------------------|

alphas = {'alpha': [0.01,0.05, 0.1, 2, 5, 25, 50, 100,500]}

grid_ridge_10 = GridSearchCV(Ridge(), alphas, cv=7, scoring='r2')
grid_ridge_25 = GridSearchCV(Ridge(), alphas, cv=7, scoring='r2')
grid_ridge_50 = GridSearchCV(Ridge(), alphas, cv=7, scoring='r2')
grid_ridge_75 = GridSearchCV(Ridge(), alphas, cv=7, scoring='r2')
grid_ridge_90 = GridSearchCV(Ridge(), alphas, cv=7, scoring='r2')

grid_ridge_10.fit(X_train_10,y_train_10)
grid_ridge_25.fit(X_train_25,y_train_25)
grid_ridge_50.fit(X_train_50,y_train_50)
grid_ridge_75.fit(X_train_75,y_train_75)
grid_ridge_90.fit(X_train_90,y_train_90)

best_ridge_model_10 = grid_ridge_10.best_estimator_
best_ridge_model_25 = grid_ridge_25.best_estimator_
best_ridge_model_50 = grid_ridge_50.best_estimator_
best_ridge_model_75 = grid_ridge_75.best_estimator_
best_ridge_model_90 = grid_ridge_90.best_estimator_

best_ridge_pred_10 = best_ridge_model_10.predict(X_test_10)
best_ridge_pred_25 = best_ridge_model_25.predict(X_test_25)
best_ridge_pred_50 = best_ridge_model_50.predict(X_test_50)
best_ridge_pred_75 = best_ridge_model_75.predict(X_test_75)
best_ridge_pred_90 = best_ridge_model_90.predict(X_test_90)

grid_ridge_eval_10 = eval_model(y_test_10,best_ridge_pred_10)
grid_ridge_eval_25 = eval_model(y_test_25,best_ridge_pred_25)
grid_ridge_eval_50 = eval_model(y_test_50,best_ridge_pred_50)
grid_ridge_eval_75 = eval_model(y_test_75,best_ridge_pred_75)
grid_ridge_eval_90 = eval_model(y_test_90,best_ridge_pred_90)

grid_ridge_rmse_10, grid_ridge_r2_10, grid_ridge_mae_10, grid_ridge_mape_10 = grid_ridge_eval_10['rmse'], grid_ridge_eval_10['r2'], grid_ridge_eval_10['mae'], grid_ridge_eval_10['mape']
grid_ridge_rmse_25, grid_ridge_r2_25, grid_ridge_mae_25, grid_ridge_mape_25 = grid_ridge_eval_25['rmse'], grid_ridge_eval_25['r2'], grid_ridge_eval_25['mae'], grid_ridge_eval_25['mape']
grid_ridge_rmse_50, grid_ridge_r2_50, grid_ridge_mae_50, grid_ridge_mape_50 = grid_ridge_eval_50['rmse'], grid_ridge_eval_50['r2'], grid_ridge_eval_50['mae'], grid_ridge_eval_50['mape']
grid_ridge_rmse_75, grid_ridge_r2_75, grid_ridge_mae_75, grid_ridge_mape_75 = grid_ridge_eval_75['rmse'], grid_ridge_eval_75['r2'], grid_ridge_eval_75['mae'], grid_ridge_eval_75['mape']
grid_ridge_rmse_90, grid_ridge_r2_90, grid_ridge_mae_90, grid_ridge_mape_90 = grid_ridge_eval_90['rmse'], grid_ridge_eval_90['r2'], grid_ridge_eval_90['mae'], grid_ridge_eval_90['mape']

print('The evaluation metrics for the best ridge model regression with a: \n')
print(f'10% training set are:\nRMSE: {grid_ridge_rmse_10:.2f}\nR squared: {grid_ridge_r2_10:.2f}\nMAE: {grid_ridge_mae_10:2f}\nMAPE: {grid_ridge_mape_10*100:.2f}%\n')
print(f'25% training set are:\nRMSE: {grid_ridge_rmse_25:.2f}\nR squared: {grid_ridge_r2_25:.2f}\nMAE: {grid_ridge_mae_25:2f}\nMAPE: {grid_ridge_mape_25*100:.2f}%\n')
print(f'50% training set are:\nRMSE: {grid_ridge_rmse_50:.2f}\nR squared: {grid_ridge_r2_50:.2f}\nMAE: {grid_ridge_mae_50:2f}\nMAPE: {grid_ridge_mape_50*100:.2f}%\n')
print(f'75% training set are:\nRMSE: {grid_ridge_rmse_75:.2f}\nR squared: {grid_ridge_r2_75:.2f}\nMAE: {grid_ridge_mae_75:2f}\nMAPE: {grid_ridge_mape_75*100:.2f}%\n')
print(f'90% training set are:\nRMSE: {grid_ridge_rmse_90:.2f}\nR squared: {grid_ridge_r2_90:.2f}\nMAE: {grid_ridge_mae_90:2f}\nMAPE: {grid_ridge_mape_90*100:.2f}%')

plot_act_v_pred(
    y_test=y_test_90,
    prediction=best_ridge_pred_90,
    x_label='Actual Renewable Share (%)',
    y_label='Predicted Renewable Share (%)',
    chart_title='Ridge Regression Performance (90% Training)',
    save_filename='rr_act_vs_pred.png',
    output_filepath=MODEL_VIS)

plot_residuals(
    y_test=y_test_90,
    prediction=best_ridge_pred_90,
    x_label='Actual Renewable Share (%)',
    y_label='Residuals',
    chart_title='Ridge Regression Residuals (90% Training)',
    save_filename='rr_residuals.png',
    output_filepath=MODEL_VIS)

# %%
#-----------------------------------------------|
#          Baseline 3: Lasso Regression         |
#-----------------------------------------------|
#
# Grid Search cross validation to find the optimal alpha (λ) 
# which controls the strength of the ridge regression hyperparamter
# to find a balance between minimizing the residual sum of squares 
# and penalizing the sum of square coefficients

alphas = {'alpha': [0.001,0.005, 0.01, 0.1, 2, 5, 50]}
maxiter = 10000000

grid_lasso_10 = GridSearchCV(Lasso(max_iter=maxiter), alphas, cv=7, scoring='r2')
grid_lasso_25 = GridSearchCV(Lasso(max_iter=maxiter), alphas, cv=7, scoring='r2')
grid_lasso_50 = GridSearchCV(Lasso(max_iter=maxiter), alphas, cv=7, scoring='r2')
grid_lasso_75 = GridSearchCV(Lasso(max_iter=maxiter), alphas, cv=7, scoring='r2')
grid_lasso_90 = GridSearchCV(Lasso(max_iter=maxiter), alphas, cv=7, scoring='r2')

grid_lasso_10.fit(X_train_10,y_train_10)
grid_lasso_25.fit(X_train_25,y_train_25)
grid_lasso_50.fit(X_train_50,y_train_50)
grid_lasso_75.fit(X_train_75,y_train_75)
grid_lasso_90.fit(X_train_90,y_train_90)

best_lasso_model_10 = grid_lasso_10.best_estimator_
best_lasso_model_25 = grid_lasso_25.best_estimator_
best_lasso_model_50 = grid_lasso_50.best_estimator_
best_lasso_model_75 = grid_lasso_75.best_estimator_
best_lasso_model_90 = grid_lasso_90.best_estimator_

best_lasso_pred_10 = best_lasso_model_10.predict(X_test_10)
best_lasso_pred_25 = best_lasso_model_25.predict(X_test_25)
best_lasso_pred_50 = best_lasso_model_50.predict(X_test_50)
best_lasso_pred_75 = best_lasso_model_75.predict(X_test_75)
best_lasso_pred_90 = best_lasso_model_90.predict(X_test_90)

grid_lasso_eval_10 = eval_model(y_test_10,best_lasso_pred_10)
grid_lasso_eval_25 = eval_model(y_test_25,best_lasso_pred_25)
grid_lasso_eval_50 = eval_model(y_test_50,best_lasso_pred_50)
grid_lasso_eval_75 = eval_model(y_test_75,best_lasso_pred_75)
grid_lasso_eval_90 = eval_model(y_test_90,best_lasso_pred_90)

grid_lasso_rmse_10, grid_lasso_r2_10, grid_lasso_mae_10, grid_lasso_mape_10 = grid_lasso_eval_10['rmse'], grid_lasso_eval_10['r2'], grid_lasso_eval_10['mae'], grid_lasso_eval_10['mape']
grid_lasso_rmse_25, grid_lasso_r2_25, grid_lasso_mae_25, grid_lasso_mape_25 = grid_lasso_eval_25['rmse'], grid_lasso_eval_25['r2'], grid_lasso_eval_25['mae'], grid_lasso_eval_25['mape']
grid_lasso_rmse_50, grid_lasso_r2_50, grid_lasso_mae_50, grid_lasso_mape_50 = grid_lasso_eval_50['rmse'], grid_lasso_eval_50['r2'], grid_lasso_eval_50['mae'], grid_lasso_eval_50['mape']
grid_lasso_rmse_75, grid_lasso_r2_75, grid_lasso_mae_75, grid_lasso_mape_75 = grid_lasso_eval_75['rmse'], grid_lasso_eval_75['r2'], grid_lasso_eval_75['mae'], grid_lasso_eval_75['mape']
grid_lasso_rmse_90, grid_lasso_r2_90, grid_lasso_mae_90, grid_lasso_mape_90 = grid_lasso_eval_90['rmse'], grid_lasso_eval_90['r2'], grid_lasso_eval_90['mae'], grid_lasso_eval_90['mape']

print('The evaluation metrics for the best lasso model regression with a: \n')
print(f'10% training set are:\nRMSE: {grid_lasso_rmse_10:.2f}\nR squared: {grid_lasso_r2_10:.2f}\nMAE: {grid_lasso_mae_10:2f}\nMAPE: {grid_lasso_mape_10*100:.2f}%\n')
print(f'25% training set are:\nRMSE: {grid_lasso_rmse_25:.2f}\nR squared: {grid_lasso_r2_25:.2f}\nMAE: {grid_lasso_mae_25:2f}\nMAPE: {grid_lasso_mape_25*100:.2f}%\n')
print(f'50% training set are:\nRMSE: {grid_lasso_rmse_50:.2f}\nR squared: {grid_lasso_r2_50:.2f}\nMAE: {grid_lasso_mae_50:2f}\nMAPE: {grid_lasso_mape_50*100:.2f}%\n')
print(f'75% training set are:\nRMSE: {grid_lasso_rmse_75:.2f}\nR squared: {grid_lasso_r2_75:.2f}\nMAE: {grid_lasso_mae_75:2f}\nMAPE: {grid_lasso_mape_75*100:.2f}%\n')
print(f'90% training set are:\nRMSE: {grid_lasso_rmse_90:.2f}\nR squared: {grid_lasso_r2_90:.2f}\nMAE: {grid_lasso_mae_90:2f}\nMAPE: {grid_lasso_mape_90*100:.2f}%')

nonzero = np.sum(best_lasso_model_90.coef_ != 0)
print(f"Non-zero Lasso coefficients: {nonzero} / {len(best_lasso_model_90.coef_)}")

plot_act_v_pred(
    y_test=y_test_90,
    prediction=best_lasso_pred_90,
    x_label='Actual Renewable Share (%)',
    y_label='Predicted Renewable Share (%)',
    chart_title='Lasso Regression Performance (90% Training)',
    save_filename='lasso_act_vs_pred.png',
    output_filepath=MODEL_VIS)

plot_residuals(
    y_test=y_test_90,
    prediction=best_lasso_pred_90,
    x_label='Actual Renewable Share (%)',
    y_label='Residuals',
    chart_title='Lasso Regression Residuals (90% Training)',
    save_filename='lasso_residuals.png',
    output_filepath=MODEL_VIS)
#%%[markdown]
#---------------------------------------------------------------------------------------
# SECTION 5 – Non-inear Regressions
#       
#---------------------------------------------------------------------------------------
#%%
#-----------------------------------------------|
#   Advanced Model 1: Random Forest Regression  |
#-----------------------------------------------|
#
param_grid_rf = {
    'n_estimators': [300, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2', None]
}

rf_model_10 = RandomForestRegressor(random_state=42)
rf_model_25 = RandomForestRegressor(random_state=42)
rf_model_50 = RandomForestRegressor(random_state=42)
rf_model_75 = RandomForestRegressor(random_state=42)
rf_model_90 = RandomForestRegressor(random_state=42)

grid_rf_10 = GridSearchCV(estimator=rf_model_10, param_grid=param_grid_rf, scoring='r2', cv=5, n_jobs=-1, verbose=1)
grid_rf_25 = GridSearchCV(estimator=rf_model_25, param_grid=param_grid_rf, scoring='r2', cv=5, n_jobs=-1, verbose=1)
grid_rf_50 = GridSearchCV(estimator=rf_model_50, param_grid=param_grid_rf, scoring='r2', cv=5, n_jobs=-1, verbose=1)
grid_rf_75 = GridSearchCV(estimator=rf_model_75, param_grid=param_grid_rf, scoring='r2', cv=5, n_jobs=-1, verbose=1)
grid_rf_90 = GridSearchCV(estimator=rf_model_90, param_grid=param_grid_rf, scoring='r2', cv=5, n_jobs=-1, verbose=1)

grid_rf_10.fit(X_train_10, y_train_10)
grid_rf_25.fit(X_train_25, y_train_25)
grid_rf_50.fit(X_train_50, y_train_50)
grid_rf_75.fit(X_train_75, y_train_75)
grid_rf_90.fit(X_train_90, y_train_90)

best_grid_rf_10 = grid_rf_10.best_estimator_
best_grid_rf_25 = grid_rf_25.best_estimator_
best_grid_rf_50 = grid_rf_50.best_estimator_
best_grid_rf_75 = grid_rf_75.best_estimator_
best_grid_rf_90 = grid_rf_90.best_estimator_

rf_model_pred_10 = best_grid_rf_10.predict(X_test_10)
rf_model_pred_25 = best_grid_rf_25.predict(X_test_25)
rf_model_pred_50 = best_grid_rf_50.predict(X_test_50)
rf_model_pred_75 = best_grid_rf_75.predict(X_test_75)
rf_model_pred_90 = best_grid_rf_90.predict(X_test_90)

rf_model_eval_10 = eval_model(y_test_10,rf_model_pred_10)
rf_model_eval_25 = eval_model(y_test_25,rf_model_pred_25)
rf_model_eval_50 = eval_model(y_test_50,rf_model_pred_50)
rf_model_eval_75 = eval_model(y_test_75,rf_model_pred_75)
rf_model_eval_90 = eval_model(y_test_90,rf_model_pred_90)

rf_rmse_10, rf_r2_10, rf_mae_10, rf_mape_10 = rf_model_eval_10['rmse'], rf_model_eval_10['r2'], rf_model_eval_10['mae'], rf_model_eval_10['mape']
rf_rmse_25, rf_r2_25, rf_mae_25, rf_mape_25 = rf_model_eval_25['rmse'], rf_model_eval_25['r2'], rf_model_eval_25['mae'], rf_model_eval_25['mape']
rf_rmse_50, rf_r2_50, rf_mae_50, rf_mape_50 = rf_model_eval_50['rmse'], rf_model_eval_50['r2'], rf_model_eval_50['mae'], rf_model_eval_50['mape']
rf_rmse_75, rf_r2_75, rf_mae_75, rf_mape_75 = rf_model_eval_75['rmse'], rf_model_eval_75['r2'], rf_model_eval_75['mae'], rf_model_eval_75['mape']
rf_rmse_90, rf_r2_90, rf_mae_90, rf_mape_90 = rf_model_eval_90['rmse'], rf_model_eval_90['r2'], rf_model_eval_90['mae'], rf_model_eval_90['mape']

print('The evaluation metrics the Random Forest Regressions with a: \n')
print(f'10% training set are:\nRMSE: {rf_rmse_10:.2f}\nR squared: {rf_r2_10:.2f}\nMAE: {rf_mae_10:2f}\nMAPE: {rf_mape_10*100:.2f}%\n')
print(f'25% training set are:\nRMSE: {rf_rmse_25:.2f}\nR squared: {rf_r2_25:.2f}\nMAE: {rf_mae_25:2f}\nMAPE: {rf_mape_25*100:.2f}%\n')
print(f'50% training set are:\nRMSE: {rf_rmse_50:.2f}\nR squared: {rf_r2_50:.2f}\nMAE: {rf_mae_50:2f}\nMAPE: {rf_mape_50*100:.2f}%\n')
print(f'75% training set are:\nRMSE: {rf_rmse_75:.2f}\nR squared: {rf_r2_75:.2f}\nMAE: {rf_mae_75:2f}\nMAPE: {rf_mape_75*100:.2f}%\n')
print(f'90% training set are:\nRMSE: {rf_rmse_90:.2f}\nR squared: {rf_r2_90:.2f}\nMAE: {rf_mae_90:2f}\nMAPE: {rf_mape_90*100:.2f}%\n')

plot_act_v_pred(
    y_test=y_test_90,
    prediction=rf_model_pred_90,
    x_label='Actual Renewable Share (%)',
    y_label='Predicted Renewable Share (%)',
    chart_title='Random Forest Performance (90% Training)',
    save_filename='rf_act_vs_pred.png',
    output_filepath=MODEL_VIS)

plot_residuals(
    y_test=y_test_90,
    prediction=rf_model_pred_90,
    x_label='Actual Renewable Share (%)',
    y_label='Residuals',
    chart_title='RF Residuals (90% Training)',
    save_filename='rf_residuals.png',
    output_filepath=MODEL_VIS)
# %%
#-----------------------------------------------|
#           Advanced Model 2: XGBoost           |
#-----------------------------------------------|
#
xgb_model_10 = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model_25 = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model_50 = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model_75 = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model_90 = XGBRegressor(objective='reg:squarederror', random_state=42)

param_grid = {
    'n_estimators': [300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.9, 1.0]
}

grid_xgb_10 = GridSearchCV(estimator=xgb_model_10, param_grid=param_grid, scoring='r2', cv=5, verbose=1, n_jobs=-1)
grid_xgb_25 = GridSearchCV(estimator=xgb_model_25, param_grid=param_grid, scoring='r2', cv=5, verbose=1, n_jobs=-1)
grid_xgb_50 = GridSearchCV(estimator=xgb_model_50, param_grid=param_grid, scoring='r2', cv=5, verbose=1, n_jobs=-1)
grid_xgb_75 = GridSearchCV(estimator=xgb_model_75, param_grid=param_grid, scoring='r2', cv=5, verbose=1, n_jobs=-1)
grid_xgb_90 = GridSearchCV(estimator=xgb_model_90, param_grid=param_grid, scoring='r2', cv=5, verbose=1, n_jobs=-1)

grid_xgb_10.fit(X_train_10, y_train_10)
grid_xgb_25.fit(X_train_25, y_train_25)
grid_xgb_50.fit(X_train_50, y_train_50)
grid_xgb_75.fit(X_train_75, y_train_75)
grid_xgb_90.fit(X_train_90, y_train_90)


xgb_pred_10 = grid_xgb_10.predict(X_test_10)
xgb_pred_25 = grid_xgb_25.predict(X_test_25)
xgb_pred_50 = grid_xgb_50.predict(X_test_50)
xgb_pred_75 = grid_xgb_75.predict(X_test_75)
xgb_pred_90 = grid_xgb_90.predict(X_test_90)

xgb_eval_10 = eval_model(y_test_10,xgb_pred_10)
xgb_eval_25 = eval_model(y_test_25,xgb_pred_25)
xgb_eval_50 = eval_model(y_test_50,xgb_pred_50)
xgb_eval_75 = eval_model(y_test_75,xgb_pred_75)
xgb_eval_90 = eval_model(y_test_90,xgb_pred_90)

xgb_rmse_10, xgb_r2_10, xgb_mae_10, xgb_mape_10 = xgb_eval_10['rmse'], xgb_eval_10['r2'], xgb_eval_10['mae'], xgb_eval_10['mape']
xgb_rmse_25, xgb_r2_25, xgb_mae_25, xgb_mape_25 = xgb_eval_25['rmse'], xgb_eval_25['r2'], xgb_eval_25['mae'], xgb_eval_25['mape']
xgb_rmse_50, xgb_r2_50, xgb_mae_50, xgb_mape_50 = xgb_eval_50['rmse'], xgb_eval_50['r2'], xgb_eval_50['mae'], xgb_eval_50['mape']
xgb_rmse_75, xgb_r2_75, xgb_mae_75, xgb_mape_75 = xgb_eval_75['rmse'], xgb_eval_75['r2'], xgb_eval_75['mae'], xgb_eval_75['mape']
xgb_rmse_90, xgb_r2_90, xgb_mae_90, xgb_mape_90 = xgb_eval_90['rmse'], xgb_eval_90['r2'], xgb_eval_90['mae'], xgb_eval_90['mape']

print('The evaluation metrics the Extreme Gradient Boosting with a: \n')
print(f'10% training set are:\nRMSE: {xgb_rmse_10:.2f}\nR squared: {xgb_r2_10:.2f}\nMAE: {xgb_mae_10:2f}\nMAPE: {xgb_mape_10*100:.2f}%\n')
print(f'25% training set are:\nRMSE: {xgb_rmse_25:.2f}\nR squared: {xgb_r2_25:.2f}\nMAE: {xgb_mae_25:2f}\nMAPE: {xgb_mape_25*100:.2f}%\n')
print(f'50% training set are:\nRMSE: {xgb_rmse_50:.2f}\nR squared: {xgb_r2_50:.2f}\nMAE: {xgb_mae_50:2f}\nMAPE: {xgb_mape_50*100:.2f}%\n')
print(f'75% training set are:\nRMSE: {xgb_rmse_75:.2f}\nR squared: {xgb_r2_75:.2f}\nMAE: {xgb_mae_75:2f}\nMAPE: {xgb_mape_75*100:.2f}%\n')
print(f'90% training set are:\nRMSE: {xgb_rmse_90:.2f}\nR squared: {xgb_r2_90:.2f}\nMAE: {xgb_mae_90:2f}\nMAPE: {xgb_mape_90*100:.2f}%\n')

plot_act_v_pred(
    y_test=y_test_90,
    prediction=xgb_pred_90,
    x_label='Actual Renewable Share (%)',
    y_label='Predicted Renewable Share (%)',
    chart_title='XGBoost Performance (90% Training)',
    save_filename='xgb_act_vs_pred.png',
    output_filepath=MODEL_VIS)

plot_residuals(
    y_test=y_test_90,
    prediction=xgb_pred_90,
    x_label='Actual Renewable Share (%)',
    y_label='Residuals',
    chart_title='XGBoost Residuals (90% Training)',
    save_filename='xgb_residuals.png',
    output_filepath=MODEL_VIS)
