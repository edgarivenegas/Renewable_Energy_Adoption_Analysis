from sklearn.metrics import root_mean_squared_error,mean_absolute_error, mean_absolute_percentage_error, r2_score
import numpy as np
import pandas as pd
from typing import Union

def eval_model(
        y_test: Union[np.ndarray, pd.Series],
        prediction: np.ndarray
):
    '''
    Evaluate a model based on the target variable and predition.

    Parameters
    ---
    `y_test`:`pd.Series`
        The correct target variable values.
    `prediction`:`np.ndarray`
        The estimated values of the target variable based on the model.
    
    Returns
    ---
    `key_metrics`:`dict`
        A dictionary with RMSE, R2, MAE, and MAPE.
        - `rmse`:`float`
        Root mean squared error is the square root of the average differences between predicted and actual values.
        - `r2`:`float`
        Coefficient of determination represents the proportion of variation in the target variable explained by the model.
        - `mae`:`float`
        Mean absolute error measures the average diffeence between predicted and actual values.
        - `mape`:`float`
        Mean absolute percentage error measures the absolute percent error between prredicted and actual values.
    
    Raises
    ---
    `ValueError`
        If `y_test` and `prediction` are different lengths, or empty.
    `TypeError`
        If `y_test` or `prediction` are not a numpy ndarray.   
    '''
    if not isinstance(y_test,pd.Series) or not isinstance(prediction, np.ndarray):
        raise TypeError('Target variable vector and prediction vector must be a numpy ndarray')

    if len(y_test) != len(prediction):
        raise ValueError('Targe variable vector and prediction vector must be the same length.')
    if len(y_test) == 0:
        raise ValueError('Target variable vector much not be empty.')
    if len(prediction) == 0:
        raise ValueError('Prediction variable vector must not be empty.')
    rmse = root_mean_squared_error(y_test,prediction)
    r2 = r2_score(y_test,prediction)
    mae = mean_absolute_error(y_test,prediction)
    mape = mean_absolute_percentage_error(y_test,prediction)

    return {'rmse':rmse,'r2':r2,'mae':mae,'mape':mape}
