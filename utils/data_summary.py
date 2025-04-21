import pandas as pd
import numpy as np
from typing import Union
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def summarize_data(
        data: pd.DataFrame
) -> None:
    '''
    Creates a summary of a pandas DataFrame and prints its shape, column datatypes, and missing values

    Parameters
    ---
    `data`: `pd.DataFrame`
        A 2D pandas Dataframe containing data.

    Returns
    ---
    `None`
        This function prints the dataset's shape, features data types, and missing values

    Raises
    ---
    `TypeError`
        If the `data` is not a pandas DataFrame.
    `ValueError`
        If the `data` is empty or not a 2D pandas DataFrame.
    
    ---
    '''
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected input `data` to be of type `pd.DataFrame`, but received `{type(data).__name__}` instead.")
    if data.empty or data.ndim != 2:
        raise ValueError('Input `data` must be non-empty and a 2D pandas DataFrame.')
    print('Shape:', data.shape)
    print('\nData Types:\n', data.dtypes)
    print('\nMissing Values:\n', data.isnull().sum())

def preprocess_pipeline(
        data: pd.DataFrame, 
        category_features: list[str], 
        quantitative_features: list[str], 
        target_label: str,
        show_features: bool=False
) -> tuple[(np.ndarray, pd.DataFrame), pd.Series]:
    '''
    Establish features and target variable and a full preprocessing pipeline using sklearn
    MinMaxScaler, StandardScaler, and OneHotEncoder from sklearn.pipeline and sklearn.preprocessing

    Parameters
    ---
    `data`: `pd.DataFrame`
        A 2D pandas Dataframe containing only numerical data, also known as the feature matrix.
    `category_features`: `list` of `str`
        List of category feature names, formatted as a list of strings.
    `quantitative_features`: `list` of `float` or `int`
        List of quantitative feature names, formatted as a list of strings.
    `target_label`: `str`
        The name of the target feature.
    `show_features: `bool`
        Whether to print a list of columns names in the feature matrix
    
    Returns
    ---
    `tuple`
        A tuple with two elements (X_processed, y):
            `X_processed`:`np.ndarray`
                The preprocessed feature array, transformed according sklearn packages.
            `y`:`pd.Series`
                The target variable as a pandas Series.

    Raises
    ---
    `ValueError`
        If the `data` is empty or not a 2D pandas DataFrame.
    `TypeError`
        If the `data` is not a pandas DataFrame or `target_label` is not a string.
    `KeyError`
        If any of the feature columns are missing from the `data`.
    `RuntimeError`
        Any unexpected errors during preprocessing.

    ---
    '''

    if data.empty or data.ndim != 2:
        raise ValueError('Input `data` must be non-empty and a 2D pandas DataFrame.')

    # Check if dataset is a pandas Dataframe, and if the  target_label is a string
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f'Expected input `data` to be of type pandas DataFrame, but received `{type(data).__name__}` instead.')
    if not isinstance(target_label, str):
        raise TypeError(f'Expected input `target_label` to be of type `str`, but received `{type(target_label).__name__}` instead.')

    try:
        # Separate the features and target variable
        X = data[category_features+quantitative_features]
        y = data[target_label]

        # Pipeline for quantitative features
        quant_pipeline_norm = Pipeline(steps=[('scaler', MinMaxScaler())])
        quant_pipeline_std = Pipeline(steps=[('scaler', StandardScaler())])

        # Pipeline for categorical features and encoding using OneHot 
        cat_pipeline = Pipeline(steps=[('encoder', OneHotEncoder(sparse_output=False))])

        # Combination pipeline for full preprocessing pipeline
        preprocessor = ColumnTransformer(
        transformers=[
            ('quant_norm', quant_pipeline_norm, quantitative_features), 
            ('quant_standardize', quant_pipeline_std, quantitative_features), 
            ('cat', cat_pipeline, category_features)])
            
        X_processed = preprocessor.fit_transform(X)

        # Column names from OneHotEncoder
        cat_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(category_features)

        # Feature names 
        feature_names = (
            [f"{col}_normalized" for col in quantitative_features] +
            [f"{col}_standardized" for col in quantitative_features] +
            list(cat_names)
        )

        # Transform dataframe into ndarray
        X_df = pd.DataFrame(X_processed, columns=feature_names)
        features_list = X_df.columns.tolist()

        print(f'Total number of features for processing: {len(features_list)}\n Number of categorical features: {len(cat_names)}\n Number of quantitative features: {2*len(quantitative_features)}')
        if show_features == True:
            print('All features:\n',' , \n'.join(features_list) )
        return X_processed, y
    
    except KeyError as ke:
        raise KeyError(f'Column error during preprocessing: {ke}. Ensure all features are present in the DataFrame.')
    except Exception as e:
        raise RuntimeError(f'An unexpected error happend during preprocessing: {type(e).__name__} - {e}')

def train_test_splits(
        X: Union[np.ndarray,pd.DataFrame], 
        y: Union[np.ndarray, pd.Series], 
        train_sizes: list[float] = [0.2,0.5,0.9], 
        random_state: int = 42):
    '''
    Split a feature matrix into training and testing subsets based on a list of floating training sizes.

    Parameters
    ---
    `X` : `np.ndarray` or `pd.DataFrame`
        A 2D ndarray or pandas DataFrame representing the feature matrix.
    `y` : `np.ndarray` or `pd.Series`
        Target vector.
    train_sizes : list of floats
        Proportion of the dataset to be used for training.
    `random_state` : `int`
        Random seed.

    Returns
    ---
    `train_test_subset` : dict
        Dictionary where each key is the train_size and the value is a 4 component tuple:
        (X_train, X_test, y_train, y_test)
    
    Raises
    ---
    `TypeError`
        If feature matrix is not `np.ndarray` or `pd.DataFrame`
    `RuntimeError`
        If an error occurs while splitting `X` into training and testing subsets.

    ---
    '''
    if not isinstance(X, Union[np.ndarray, pd.DataFrame]):
        raise TypeError(f'Expected input `X` to be type ndarray or DataFrame, but received {type(X).__name__} instead.')
    try:
        train_test_subset = {}
        for split in train_sizes:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=random_state)
            train_test_subset[split] = (X_train, X_test, y_train, y_test)
        return train_test_subset

    except Exception as e:
        raise RuntimeError(f'An error occurred while splitting the feature matrix into training and testing subsets: {e}')

def linear_reg_pred(X_train: Union[np.ndarray,pd.DataFrame], 
            X_test: Union[np.ndarray,pd.DataFrame], 
            y_train: Union[np.ndarray,pd.Series], 
) -> np.ndarray:
    '''
    Train baseline1 linear regression model using training data 
    to predict test data, and returns prediction valules.

    Parameters
    ---
    `X_train` : `np.ndarray` or `pd.DataFrame`
        Training feature matrix.
    `X_test` : `np.ndarray` or `pd.DataFrame`
        Test feature matrix.
    `y_train` : `np.ndarray` or `pd.Series`
        Target variable for training.

    Returns
    ---
    `lr_pred`
        The predicted values based on the input variables.
    '''
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    lr_pred = lr_model.predict(X_test)    

    return lr_pred