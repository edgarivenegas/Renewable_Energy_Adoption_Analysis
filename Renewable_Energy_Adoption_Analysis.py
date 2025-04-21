#%% [markdown]
## Predicting Renewable Energy Growth Using Machine Learning and Policy Analysis
# ---
#
# **Author:** Edgar Venegas 
#
# **Estimated Duration:** 40 hours 
#
# **Elapsed Time:** 5+7+5+3+3+2+5 hours  
# 
# ---
# #### **Objective:** 
# Use machine learning techniques (XGBoost, Random Forest, LSTM) and natural language processing (NLP) Hugging Face Transformers to predict countries' future renewable energy share using infrastructure investment information, annual energy production, capacity, and environmental impact data.
# 
# ---
# #### **Tools & Libraries**  
# - **Data Handling:** pandas, numpy  
# - **Visualization:** matplotlib, seaborn  
# - **Machine Learning:** scikit-learn, XGBoost, PyTorch  
# - **NLP:** Hugging Face Transformers
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
from xgboost import XGBRegressor
from utils import plot_linechart, plot_correlations, plot_rsquared, plot_act_v_pred, plot_residuals
from utils import summarize_data, preprocess_pipeline, train_test_splits, linear_reg_pred
from utils import eval_model


# Set up project folders
base_dir = Path.cwd()
out_dir = base_dir / 'output'
LINECHART_DIR = out_dir / 'line_charts'
HEATMAP_DIR = out_dir / 'heat_maps'

for path in [LINECHART_DIR, HEATMAP_DIR]:
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
        LINECHART_DIR, 
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
    output_filepath=HEATMAP_DIR,
    annotate=True
)

plot_rsquared(
    data=df,
    chart_title='Coefficient of Determination Heatmap',
    cmap='coolwarm',
    figsize=(24,22),
    save_filename='rsquared_heatmap.png',
    output_filepath=HEATMAP_DIR,
    annotate=True
)
#%%[markdown]
#---------------------------------------------------------------------------------------
# SECTION 3 – Normalization, Standardization, and Policy Analysis
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
train_sizes = [0.1,0.2,0.50,0.75,0.9]
train_test_dataset = train_test_splits(X_processed,y,train_sizes)

#            ---------------Insert Policy Analysis---------------
'''Get major policy changes throughout the time frame for each year, and for eachcountry, and relate everything to that timeperiod with a binary before or after statistic
to then incorporate in the model (i.e. 5 years before a positive renewable growth policy change there is X% renewable share growth, now there is X+23% growth)
'''
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
X_train_20, X_test_20, y_train_20, y_test_20 = train_test_dataset[0.2][0] ,train_test_dataset[0.2][1],train_test_dataset[0.2][2], train_test_dataset[0.2][3]

# Training set 50% -- Testing Set 50%
X_train_50, X_test_50, y_train_50, y_test_50 = train_test_dataset[0.5][0] ,train_test_dataset[0.5][1],train_test_dataset[0.5][2], train_test_dataset[0.5][3]

# Training set 75% -- Testing Set 25%
X_train_75, X_test_75, y_train_75, y_test_75 = train_test_dataset[0.75][0] ,train_test_dataset[0.75][1],train_test_dataset[0.75][2], train_test_dataset[0.75][3]

# Training set 90% -- Testing Set 10%
X_train_90, X_test_90, y_train_90, y_test_90 = train_test_dataset[0.9][0] ,train_test_dataset[0.9][1],train_test_dataset[0.9][2], train_test_dataset[0.9][3]

#%%
lr_pred_90 = linear_reg_pred(X_train_10, X_test_10, y_train_10)
lr_pred_75 = linear_reg_pred(X_train_20, X_test_20, y_train_20)
lr_pred_50 = linear_reg_pred(X_train_50, X_test_50, y_train_50)
lr_pred_20 = linear_reg_pred(X_train_75, X_test_75, y_train_75)
lr_pred_10 = linear_reg_pred(X_train_90, X_test_90, y_train_90)

# Linear Regression Evaluation
lr_eval_10 = eval_model(y_test_10,lr_pred_90)
lr_eval_20 = eval_model(y_test_20,lr_pred_75)
lr_eval_50 = eval_model(y_test_50,lr_pred_50)
lr_eval_75 = eval_model(y_test_75,lr_pred_20)
lr_eval_90 = eval_model(y_test_90,lr_pred_10)

lr_rmse_10, lr_r2_10, lr_mae_10, lr_mape_10 = lr_eval_10['rmse'], lr_eval_10['r2'], lr_eval_10['mae'], lr_eval_10['mape']
lr_rmse_20, lr_r2_20, lr_mae_20, lr_mape_20 = lr_eval_20['rmse'], lr_eval_20['r2'], lr_eval_20['mae'], lr_eval_20['mape']
lr_rmse_50, lr_r2_50, lr_mae_50, lr_mape_50 = lr_eval_50['rmse'], lr_eval_50['r2'], lr_eval_50['mae'], lr_eval_50['mape']
lr_rmse_75, lr_r2_75, lr_mae_75, lr_mape_75 = lr_eval_75['rmse'], lr_eval_75['r2'], lr_eval_75['mae'], lr_eval_75['mape']
lr_rmse_90, lr_r2_90, lr_mae_90, lr_mape_90 = lr_eval_90['rmse'], lr_eval_90['r2'], lr_eval_90['mae'], lr_eval_90['mape']


print('The evaluation metrics for the linear regression with a: \n')
print(f'10% training set are:\nRMSE: {lr_rmse_10:.2f}\nR squared: {lr_r2_10:.2f}\nMAE: {lr_mae_10:2f}\nMAPE: {lr_mape_10*100:.2f}%\n')
print(f'20% training set are:\nRMSE: {lr_rmse_20:.2f}\nR squared: {lr_r2_20:.2f}\nMAE: {lr_mae_20:2f}\nMAPE: {lr_mape_20*100:.2f}%\n')
print(f'50% training set are:\nRMSE: {lr_rmse_50:.2f}\nR squared: {lr_r2_50:.2f}\nMAE: {lr_mae_50:2f}\nMAPE: {lr_mape_50*100:.2f}%\n')
print(f'75% training set are:\nRMSE: {lr_rmse_75:.2f}\nR squared: {lr_r2_75:.2f}\nMAE: {lr_mae_75:2f}\nMAPE: {lr_mape_75*100:.2f}%\n')
print(f'90% training set are:\nRMSE: {lr_rmse_90:.2f}\nR squared: {lr_r2_90:.2f}\nMAE: {lr_mae_90:2f}\nMAPE: {lr_mape_90*100:.2f}%')


#-----------------------------------------------|
#     Baseline 1: Linear Regression Visuals     |
#-----------------------------------------------|
#
# Plot 10% testing Set
plot_act_v_pred(
    y_test=y_test_10,
    prediction=lr_pred_90,
    x_label='Actual Renewable Energy Share (%)',
    y_label='Predicted Renewable Energy Share (%)',
    chart_title='Linear Regression Performance: Training set 10%',
    xtick_size=12,
    ytick_size=12,
    chart_title_size=18,
    output_filepath=LINECHART_DIR)
plot_residuals(
    y_test_10,
    lr_pred_90,
    x_label='Actual Renewable Energy Share (%)',
    y_label='Predicted Renewable Energy Share (%)',
    chart_title='Linear Regression Performance: Training set 10%',
    xtick_size=12,
    ytick_size=12,
    chart_title_size=18,
    output_filepath=LINECHART_DIR)
#
# Plot 10% testing Set
plot_act_v_pred(
    y_test=y_test_20,
    prediction=lr_pred_90,
    x_label='Actual Renewable Energy Share (%)',
    y_label='Predicted Renewable Energy Share (%)',
    chart_title='Linear Regression Performance: Training set 10%',
    xtick_size=12,
    ytick_size=12,
    chart_title_size=18,
    output_filepath=LINECHART_DIR)
plot_residuals(
    y_test_10,
    lr_pred_90,
    x_label='Actual Renewable Energy Share (%)',
    y_label='Predicted Renewable Energy Share (%)',
    chart_title='Linear Regression Performance: Training set 10%',
    xtick_size=12,
    ytick_size=12,
    chart_title_size=18,
    output_filepath=LINECHART_DIR)

# %%
#-----------------------------------------------|
#          Baseline 2: Ridge Regression         |
#-----------------------------------------------|

alphas = {'alpha': [0.01,0.05, 0.1, 2, 5, 25, 50, 100,500]}

grid_ridge_10 = GridSearchCV(Ridge(), alphas, cv=7, scoring='neg_mean_squared_error')
grid_ridge_20 = GridSearchCV(Ridge(), alphas, cv=7, scoring='neg_mean_squared_error')
grid_ridge_50 = GridSearchCV(Ridge(), alphas, cv=7, scoring='neg_mean_squared_error')
grid_ridge_75 = GridSearchCV(Ridge(), alphas, cv=7, scoring='neg_mean_squared_error')
grid_ridge_90 = GridSearchCV(Ridge(), alphas, cv=7, scoring='neg_mean_squared_error')

grid_ridge_10.fit(X_train_10,y_train_10)
grid_ridge_20.fit(X_train_20,y_train_20)
grid_ridge_50.fit(X_train_50,y_train_50)
grid_ridge_75.fit(X_train_75,y_train_75)
grid_ridge_90.fit(X_train_90,y_train_90)

best_ridge_model_10 = grid_ridge_10.best_estimator_
best_ridge_model_20 = grid_ridge_20.best_estimator_
best_ridge_model_50 = grid_ridge_50.best_estimator_
best_ridge_model_75 = grid_ridge_75.best_estimator_
best_ridge_model_90 = grid_ridge_90.best_estimator_

best_ridge_pred_10 = best_ridge_model_10.predict(X_test_10)
best_ridge_pred_20 = best_ridge_model_20.predict(X_test_20)
best_ridge_pred_50 = best_ridge_model_50.predict(X_test_50)
best_ridge_pred_75 = best_ridge_model_75.predict(X_test_75)
best_ridge_pred_90 = best_ridge_model_90.predict(X_test_90)

grid_ridge_eval_10 = eval_model(y_test_10,best_ridge_pred_10)
grid_ridge_eval_20 = eval_model(y_test_20,best_ridge_pred_20)
grid_ridge_eval_50 = eval_model(y_test_50,best_ridge_pred_50)
grid_ridge_eval_75 = eval_model(y_test_75,best_ridge_pred_75)
grid_ridge_eval_90 = eval_model(y_test_90,best_ridge_pred_90)

grid_ridge_rmse_10, grid_ridge_r2_10, grid_ridge_mae_10, grid_ridge_mape_10 = grid_ridge_eval_10['rmse'], grid_ridge_eval_10['r2'], grid_ridge_eval_10['mae'], grid_ridge_eval_10['mape']
grid_ridge_rmse_20, grid_ridge_r2_20, grid_ridge_mae_20, grid_ridge_mape_20 = grid_ridge_eval_20['rmse'], grid_ridge_eval_20['r2'], grid_ridge_eval_20['mae'], grid_ridge_eval_20['mape']
grid_ridge_rmse_50, grid_ridge_r2_50, grid_ridge_mae_50, grid_ridge_mape_50 = grid_ridge_eval_50['rmse'], grid_ridge_eval_50['r2'], grid_ridge_eval_50['mae'], grid_ridge_eval_50['mape']
grid_ridge_rmse_75, grid_ridge_r2_75, grid_ridge_mae_75, grid_ridge_mape_75 = grid_ridge_eval_75['rmse'], grid_ridge_eval_75['r2'], grid_ridge_eval_75['mae'], grid_ridge_eval_75['mape']
grid_ridge_rmse_90, grid_ridge_r2_90, grid_ridge_mae_90, grid_ridge_mape_90 = grid_ridge_eval_90['rmse'], grid_ridge_eval_90['r2'], grid_ridge_eval_90['mae'], grid_ridge_eval_90['mape']

print('The evaluation metrics for the best ridge model regression with a: \n')
print(f'10% training set are:\nRMSE: {grid_ridge_rmse_10:.2f}\nR squared: {grid_ridge_r2_10:.2f}\nMAE: {grid_ridge_mae_10:2f}\nMAPE: {grid_ridge_mape_10*100:.2f}%\n')
print(f'20% training set are:\nRMSE: {grid_ridge_rmse_20:.2f}\nR squared: {grid_ridge_r2_20:.2f}\nMAE: {grid_ridge_mae_20:2f}\nMAPE: {grid_ridge_mape_20*100:.2f}%\n')
print(f'50% training set are:\nRMSE: {grid_ridge_rmse_50:.2f}\nR squared: {grid_ridge_r2_50:.2f}\nMAE: {grid_ridge_mae_50:2f}\nMAPE: {grid_ridge_mape_50*100:.2f}%\n')
print(f'75% training set are:\nRMSE: {grid_ridge_rmse_75:.2f}\nR squared: {grid_ridge_r2_75:.2f}\nMAE: {grid_ridge_mae_75:2f}\nMAPE: {grid_ridge_mape_75*100:.2f}%\n')
print(f'90% training set are:\nRMSE: {grid_ridge_rmse_90:.2f}\nR squared: {grid_ridge_r2_90:.2f}\nMAE: {grid_ridge_mae_90:2f}\nMAPE: {grid_ridge_mape_90*100:.2f}%')

# %%
#-----------------------------------------------|
#          Baseline 3: Lasso Regression         |
#-----------------------------------------------|
#
# Grid Search cross validation to find the optimal alpha (λ) 
# which controls the strength of the ridge regression hyperparamter
# to find a balance between minimizing the residual sum of squares 
# and penalizing the sum of square coefficients

alphas = {'alpha': [0.01,0.05, 0.1, 2, 5, 25, 50, 100,500]}
maxiter = 10000000

grid_lasso_10 = GridSearchCV(Lasso(max_iter=maxiter), alphas, cv=7, scoring='neg_mean_squared_error')
grid_lasso_20 = GridSearchCV(Lasso(max_iter=maxiter), alphas, cv=7, scoring='neg_mean_squared_error')
grid_lasso_50 = GridSearchCV(Lasso(max_iter=maxiter), alphas, cv=7, scoring='neg_mean_squared_error')
grid_lasso_75 = GridSearchCV(Lasso(max_iter=maxiter), alphas, cv=7, scoring='neg_mean_squared_error')
grid_lasso_90 = GridSearchCV(Lasso(max_iter=maxiter), alphas, cv=7, scoring='neg_mean_squared_error')

grid_lasso_10.fit(X_train_10,y_train_10)
grid_lasso_20.fit(X_train_20,y_train_20)
grid_lasso_50.fit(X_train_50,y_train_50)
grid_lasso_75.fit(X_train_75,y_train_75)
grid_lasso_90.fit(X_train_90,y_train_90)

best_lasso_model_10 = grid_lasso_10.best_estimator_
best_lasso_model_20 = grid_lasso_20.best_estimator_
best_lasso_model_50 = grid_lasso_50.best_estimator_
best_lasso_model_75 = grid_lasso_75.best_estimator_
best_lasso_model_90 = grid_lasso_90.best_estimator_

best_lasso_pred_10 = best_lasso_model_10.predict(X_test_10)
best_lasso_pred_20 = best_lasso_model_20.predict(X_test_20)
best_lasso_pred_50 = best_lasso_model_50.predict(X_test_50)
best_lasso_pred_75 = best_lasso_model_75.predict(X_test_75)
best_lasso_pred_90 = best_lasso_model_90.predict(X_test_90)

grid_lasso_eval_10 = eval_model(y_test_10,best_lasso_pred_10)
grid_lasso_eval_20 = eval_model(y_test_20,best_lasso_pred_20)
grid_lasso_eval_50 = eval_model(y_test_50,best_lasso_pred_50)
grid_lasso_eval_75 = eval_model(y_test_75,best_lasso_pred_75)
grid_lasso_eval_90 = eval_model(y_test_90,best_lasso_pred_90)

grid_lasso_rmse_10, grid_lasso_r2_10, grid_lasso_mae_10, grid_lasso_mape_10 = grid_lasso_eval_10['rmse'], grid_lasso_eval_10['r2'], grid_lasso_eval_10['mae'], grid_lasso_eval_10['mape']
grid_lasso_rmse_20, grid_lasso_r2_20, grid_lasso_mae_20, grid_lasso_mape_20 = grid_lasso_eval_20['rmse'], grid_lasso_eval_20['r2'], grid_lasso_eval_20['mae'], grid_lasso_eval_20['mape']
grid_lasso_rmse_50, grid_lasso_r2_50, grid_lasso_mae_50, grid_lasso_mape_50 = grid_lasso_eval_50['rmse'], grid_lasso_eval_50['r2'], grid_lasso_eval_50['mae'], grid_lasso_eval_50['mape']
grid_lasso_rmse_75, grid_lasso_r2_75, grid_lasso_mae_75, grid_lasso_mape_75 = grid_lasso_eval_75['rmse'], grid_lasso_eval_75['r2'], grid_lasso_eval_75['mae'], grid_lasso_eval_75['mape']
grid_lasso_rmse_90, grid_lasso_r2_90, grid_lasso_mae_90, grid_lasso_mape_90 = grid_lasso_eval_90['rmse'], grid_lasso_eval_90['r2'], grid_lasso_eval_90['mae'], grid_lasso_eval_90['mape']

print('The evaluation metrics for the best lasso model regression with a: \n')
print(f'10% training set are:\nRMSE: {grid_lasso_rmse_10:.2f}\nR squared: {grid_lasso_r2_10:.2f}\nMAE: {grid_lasso_mae_10:2f}\nMAPE: {grid_lasso_mape_10*100:.2f}%\n')
print(f'20% training set are:\nRMSE: {grid_lasso_rmse_20:.2f}\nR squared: {grid_lasso_r2_20:.2f}\nMAE: {grid_lasso_mae_20:2f}\nMAPE: {grid_lasso_mape_20*100:.2f}%\n')
print(f'50% training set are:\nRMSE: {grid_lasso_rmse_50:.2f}\nR squared: {grid_lasso_r2_50:.2f}\nMAE: {grid_lasso_mae_50:2f}\nMAPE: {grid_lasso_mape_50*100:.2f}%\n')
print(f'75% training set are:\nRMSE: {grid_lasso_rmse_75:.2f}\nR squared: {grid_lasso_r2_75:.2f}\nMAE: {grid_lasso_mae_75:2f}\nMAPE: {grid_lasso_mape_75*100:.2f}%\n')
print(f'90% training set are:\nRMSE: {grid_lasso_rmse_90:.2f}\nR squared: {grid_lasso_r2_90:.2f}\nMAE: {grid_lasso_mae_90:2f}\nMAPE: {grid_lasso_mape_90*100:.2f}%')

#%%[markdown]
#---------------------------------------------------------------------------------------
# SECTION 5 – Non-inear Regressions
#       
#---------------------------------------------------------------------------------------
#%%
#-----------------------------------------------|
#      Model 1: Random Forest Regression        |
#-----------------------------------------------|