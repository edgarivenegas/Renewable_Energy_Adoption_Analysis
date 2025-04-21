import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_linechart(
        data: pd.DataFrame, 
        x_name: str = 'x-axis label',
        y_name: str = 'y-axis label',
        grouping_name: str = 'grouping label',
        figsize: tuple[float | int, float | int] = (12,8),
        chart_title: str='Chart Title',
        save_filename: str='linechart.png',
        output_filepath: str='output',
        fig_num:float | int=1
) -> None:
    
    '''
    Generate and save a line chart visualization from a pandas DataFrame.

    Parameters
    ---
    `data`: `pd.DataFrame`
        A 2D pandas Dataframe containing any only numeric data.
    `x_name`:`str`
        Name of the independent feature, and present in the dataset
    `y_name`:`str`
        Name of the dependent feature, and present in the dataset
    `grouping_name`: `str`
        Name of the column use as the hue, or grouping of the dataset
    `figuresize`:`tuple`
        Size of the figure, formatted as (width, height)
    `chart_title`:`str`
        The Title name of the visualization
    `save_filename`:`str`
        Name of the file to save to the directory.
    `output_filepath`:`str`
        Name of the output file path to the directory.
    `fig_num`:'float` or `int`

    Returns
    ---
    `None`
        This function generates and saves a figure to the 'output_filepath' directory and closes the plot.
    
    Raises
    ---
    `TypeError`
        When arguments `x_name`, `y_name`, `grouping_name` are not `str` data types.
        When `data` and `figuresize` are not a DataFrame and tuple respectively.
    `ValueError`
        When data an empty non-2D pandas DataFrame

    ---

    '''
    if data.empty or data.ndim != 2:
        raise ValueError('Input `data` must be non-empty and a 2D pandas DataFrame.')
    # Ensure columns are in the DataFrame
    for col in [x_name, y_name, grouping_name]:
        if col not in data.columns:
            raise ValueError(f"The column '{col}' does not exist in the dataset.")
    
    # Check that x,y, and grouping names are strings
    for arg, name in zip([x_name, y_name, grouping_name,chart_title,save_filename], ['x_name', 'y_name', 'grouping_name','chart_title','save_filename']):
        if not isinstance(arg, str):
            raise TypeError(f"Expected '{name}' to be of type 'str', but received '{type(arg).__name__}'.")
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected input {data} to be of type 'pd.DataFrame', but received {type(data).__name__} instead.")
    # Ensure figure size is in tuple format
    if not isinstance(figsize,tuple):
        raise TypeError(f"Expected input {figsize} to be of type 'tuple', but received {type(figsize).__name__}instead.")

    

    df, x_label, y_label, grouping_label = data,x_name,y_name,grouping_name

    x_mean = df[y_label].mean()
    x_std = df[y_label].std()
    cap_y_label = y_label.lower()+"'s"

    over_limit = (df[y_label] > x_mean + 1.645 * x_std)
    under_limit = (df[y_label] < x_mean - 1.645 * x_std)

    outliers_over = df[over_limit]
    outliers_under = df[under_limit]
    outliers = df[over_limit | under_limit]

    outliers_count = outliers.groupby(grouping_label).size().reset_index(name='Number of Outliers')
    outliers_count_sorted = outliers_count.sort_values(by='Number of Outliers', ascending=False).reset_index(drop=True)

    outliers_over_count = outliers_over.groupby(grouping_label).size().reset_index(name='Number of Outliers over the 95th percentile').sort_values('Number of Outliers over the 95th percentile',ascending=False)
    outliers_under_count = outliers_under.groupby(grouping_label).size().reset_index(name='Number of Outliers under the 5th percentile').sort_values('Number of Outliers under the 5th percentile',ascending=False)
  
    sns.set_style(style='white')
    plt.figure(figsize=figsize)
    sns.lineplot(x=x_label,y=y_label,hue=grouping_label,data=df,err_style=None)

    plt.subplots_adjust(bottom=.2)
    if not outliers_count_sorted.empty and not outliers_over_count.empty and not outliers_under_count.empty:
        summary_text = (
            f'Fig. {fig_num}: The {grouping_label} with the most {cap_y_label} outside of the central 90% of the dataset is \n'
            f'{outliers_count_sorted.iloc[0,0]}, with {outliers_count_sorted.iloc[0,1]} outliers. '
            f'Of these, {outliers_over_count.iloc[0,1]} over the 90th percentile, '
            f'while {outliers_under_count.iloc[0,1]} were short of the 10th percentile dataset distribution.'
        )
    else:
        summary_text = (
            f'Fig. {fig_num}: No significant groups of outliers detected for {cap_y_label} \n '
            f'beyond the central 90% of the dataset grouped by {grouping_label}.'
    )

    plt.figtext(0.5, 0.1, summary_text, ha='center', va='top', fontsize=12)

    ax = sns.scatterplot(x=outliers[x_label],y=df[y_label],legend=False)
    sns.scatterplot(x=outliers[x_label],y=outliers[y_label],hue=df[grouping_label],marker='o',s=100,legend=False,ax=ax)

    plt.axhline(y=x_mean, color='Black', linestyle='-', label='Mean', linewidth=2)
    plt.axhline(y=x_mean + x_std, color='grey', linestyle='--', label='1 Std Dev', linewidth=2)
    plt.axhline(y=x_mean - x_std, color='grey', linestyle='--', label='1 Std Dev', linewidth=2)
    plt.axhline(y=x_mean + 1.645*x_std, color='lightgrey', linestyle='-', label='90th Percentile', linewidth=2)
    plt.axhline(y=x_mean - 1.645*x_std, color='lightgrey', linestyle='-', label='10th Percentile', linewidth=2)

    plt.title(label=chart_title,fontsize=16)
    plt.xlabel(x_label,fontsize=12)
    plt.ylabel(y_label,fontsize=12)
    plt.legend(title=grouping_label,title_fontsize=12,loc='center left',fontsize=10,frameon=False,bbox_to_anchor=(.9995,.45))
    plt.subplots_adjust(right=0.85) 

    os.makedirs(output_filepath, exist_ok=True)
    plt.savefig(os.path.join(output_filepath, save_filename), format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_correlations(
    data: pd.DataFrame,
    chart_title: str='Correlation Heatmap',
    cmap: str = 'coolwarm',
    xtick_size: int=24,
    ytick_size: int=24,
    chart_title_size: int=48,
    figsize: tuple[float | int, float | int] = (12,8),
    save_filename: str = 'correlation_heatmap.png',
    output_filepath: str = 'outputs',
    annotate: bool = True
) -> None:  
    '''
    Generate and save a Seaborn correlation heatmap visualization from a pandas Dataframe
    
    Parameters
    ---
    `data`: `pd.DataFrame`
        A 2D pandas Dataframe containing any only numeric data.
    `chart_title`: `str`
        The title of the heatmap figure.
    `cmap`: `str`
        Colormap used for the heatmap
    `xtick_size`: `int`
        x-axis tick font size
    `ytick_size`: `int`
        y-axis tick font size
    `chart_title_size`: `int
        chart title font size
    `figsize`: `tuple` of `float` or `int`
        Figure dimensions in inches, formatted as (width, height)
    `save_filename`: `str`
        Output file name
    `output_filepath`: `str`
        Output file save directory 
    `fig_num`: `float` or `int`
        The figure number
    `annotate` : `bool`
        Whether to display numeric values in each cell of the heatmap 

    Returns
    ---
    `None`
        This function generates and saves a figure to the 'output_filepath' directory and closes the plot.

    Raises
    ---
    `ValueError`
        If `data` is empty or not a 2D DataFrame.
    `FileNotFoundError`
        If the `output_filepath` directory cannot be generated or accessed.
    
    ---
    '''
    if data.empty or data.ndim != 2:
        raise ValueError('Input `data` must be non-empty and a 2D pandas `DataFrame`.')
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected input {data} to be of type 'pd.DataFrame', but received {type(data).__name__} instead.")
    try:
        corr_features = []
        for feature in data.columns:
            if data[feature].dtype != 'object':
                corr_features.append(feature)
        corr_matrix = data[corr_features].corr()
        plt.figure(figsize=figsize)
        ax = sns.heatmap(corr_matrix,annot=annotate,cmap=cmap)
        colorbar = ax.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=20)
        plt.title(chart_title,fontsize=chart_title_size)
        plt.yticks(fontsize=xtick_size)
        plt.xticks(fontsize=ytick_size)
        plt.tight_layout()
        plt.savefig(output_filepath / save_filename, dpi=300)
        plt.show()
        plt.close()

    except Exception as e:
        raise FileNotFoundError(f'An error occurred while saving the heatmap: {e}')

def plot_rsquared(
    data: pd.DataFrame,
    chart_title: str='Correlation Heatmap',
    cmap: str = 'coolwarm',
    xtick_size: int=18,
    ytick_size: int=18,
    chart_title_size: int=24,
    figsize: tuple[float | int, float | int] = (12,8),
    save_filename: str = 'rsquared_heatmap.png',
    output_filepath: str = 'outputs',
    annotate: bool = True
) -> None:  
    '''
    Generate and save a Seaborn correlation heatmap visualization from a pandas Dataframe
    
    Parameters
    ---
    `data`: `pd.DataFrame`
        A 2D pandas Dataframe containing any only numeric data.
    `chart_title`: `str`
        The title of the heatmap figure.
    `cmap`: `str`
        Colormap used for the heatmap
    `xtick_size`: `int`
        x-axis tick font size
    `ytick_size`: `int`
        y-axis tick font size
    `chart_title_size`: `int
        chart title font size
    `figsize`: `tuple` of `float` or `int`
        Figure dimensions in inches, formatted as (width, height)
    `save_filename`: `str`
        Output file name
    `output_filepath`: `str`
        Output file save directory 
    `fig_num`: `float` or `int`
        The figure number
    `annotate` : `bool`
        Whether to display numeric values in each cell of the heatmap 

    Returns
    ---
    `None`
        This function generates and saves a figure to the 'output_filepath' directory and closes the plot.

    Raises
    ---
    `ValueError`
        If `data` is empty or not a 2D DataFrame.
    `FileNotFoundError`
        If the `output_filepath` directory cannot be generated or accessed.
    
    ---
    '''
    if data.empty or data.ndim != 2:
        raise ValueError('Input `data` must be non-empty and a 2D pandas `DataFrame`.')
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected input {data} to be of type 'pd.DataFrame', but received {type(data).__name__} instead.")
    try:
        corr_features = []
        for feature in data.columns:
            if data[feature].dtype != 'object':
                corr_features.append(feature)
        corr_matrix = data[corr_features].corr()
        plt.figure(figsize=figsize)
        ax = sns.heatmap(corr_matrix**2,annot=annotate,cmap=cmap)
        colorbar = ax.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=20)
        plt.title(chart_title,fontsize=chart_title_size)
        plt.yticks(fontsize=xtick_size)
        plt.xticks(fontsize=ytick_size)
        plt.tight_layout()
        plt.savefig(output_filepath / save_filename, dpi=300)
        plt.show()
        plt.close()

    except Exception as e:
        raise FileNotFoundError(f'An error occurred while saving the heatmap: {e}')

def plot_act_v_pred(
        y_test: pd.Series, 
        prediction: np.ndarray,
        x_label: str='Actual Values',
        y_label: str='Predicted Values',
        chart_title: str='Actual vs Predicted Values',
        xtick_size: int=18,
        ytick_size: int=18,
        chart_title_size: int=24,
        alpha: float=0.7,
        figsize: tuple[float | int, float | int] = (12,8),
        save_filename: str = 'actual_vs_predicted_scatter.png',
        output_filepath: str = 'outputs',
        show_grid: bool=True
) -> None:
    '''
    Plot actual and predicted values with a line of best fit.

    Parameters
    ---
    `y_test` : `pd.Series`
        The actual target feature formatted as a `pd.DataFrame`.
    `prediction` : `np.ndarray`
        The predicted target feature formatted as a `np.ndarray`,
    `x_label` : `str`
        Name of the plot x-axis.
    `y_label` : `str`
        Name of the plot y-axis.
    `xtick_size`: `int`
        x-axis tick font size
    `ytick_size`: `int`
        y-axis tick font size
    `chart_title_size`: `int
        chart title font size
    `alpha` : `float`
        Transperancy of each target value.
    `figsize`: `tuple` of `float` or `int`
        Figure dimensions in inches, formatted as (width, height)
    `save_filename` : `str`
        Filename to be saved in a directory
    `output_filepath`: `str`
        Output file save directory 
    `show_grid` : `bool`
        Whether the grid is shown.

    Returns
    ---
    `None`
        This function generates and saves a figure to the 'output_filepath' directory and closes the plot.

    Raises
    ---
    `ValueError`
        If the `y_test` is empty or not 1D
    `TypeError`
        Expected input `y_test` and `predictions` to be of data type `pd.Series` and `np.array` respectively.
    '''
    # Raise Type Errors if entry data types dont match
    if y_test.empty or y_test.ndim != 1:
        raise ValueError(f'Expected input `y_test` to non-empty and 1D `pd.DataFrame`')
    if not isinstance(y_test, pd.Series):
        raise TypeError(f"Expected input {y_test} to be of type 'pd.Series', but received {type(y_test).__name__} instead.")
    if not isinstance(prediction, np.ndarray):
        raise TypeError(f"Expected input {prediction} to be of type 'np.ndarry', but received {type(prediction).__name__} instead.")
    if not isinstance(alpha,float):
        raise TypeError
    if not isinstance(figsize, tuple):
        raise TypeError(f" Expected input {figsize} to be of type 'tuple', but received {type(figsize).__name__} instead.")

    plt.figure(figsize=figsize)
    plt.scatter(y_test, prediction, alpha=alpha)
    plt.xlabel(x_label,fontsize=16)
    plt.ylabel(y_label,fontsize=16)
    plt.plot([y_test.min(),y_test.max()],[prediction.min(),prediction.max()],'r--')
    plt.title(chart_title,fontsize=chart_title_size)
    plt.yticks(fontsize=xtick_size)
    plt.xticks(fontsize=ytick_size)
    plt.grid(show_grid)
    plt.tight_layout()
    plt.savefig(output_filepath / save_filename, dpi=300)
    plt.show()
    plt.close()

def plot_residuals(
        y_test: pd.Series, 
        prediction: np.ndarray,
        x_label: str='Predicted y',
        y_label: str='Residuals',
        chart_title: str='Residuals Plot',
        xtick_size: int=16,
        ytick_size: int=16,
        chart_title_size: int=24,
        alpha: float=0.7,
        figsize: tuple[float | int, float | int] = (12,8),
        save_filename: str = 'residuals_scatter.png',
        output_filepath: str = 'outputs',
        show_grid: bool=True
) -> None:
    '''
    Plot actual and predicted values with a line of best fit.

    Parameters
    ---
    `y_test` : `pd.Series`
        The actual target feature formatted as a `pd.DataFrame`.
    `prediction` : `np.ndarray`
        The predicted target feature formatted as a `np.ndarray`,
    `x_label` : `str`
        Name of the plot x-axis.
    `y_label` : `str`
        Name of the plot y-axis.
    `xtick_size`: `int`
        x-axis tick font size
    `ytick_size`: `int`
        y-axis tick font size
    `chart_title_size`: `int
        chart title font size
    `chart_title` : `str`
        Name of the chart title.
    `alpha` : `float`
        Transperancy of each target value.
    `figsize`: `tuple` of `float` or `int`
        Figure dimensions in inches, formatted as (width, height)
    `save_filename` : `str`
        Filename to be saved in a directory
    `output_filepath`: `str`
        Output file save directory 
    `show_grid` : `bool`
        Whether the grid is shown.

    Returns
    ---
    `None`
        This function generates and saves a figure to the 'output_filepath' directory and closes the plot.

    Raises
    ---
    `ValueError`
        If the `y_test` is empty or not 1D
    `TypeError`
        Expected input `y_test` and `predictions` to be of data type `pd.Series` and `np.array` respectively.
    '''
    # Raise Type Errors if entry data types dont match
    if y_test.empty or y_test.ndim != 1:
        raise ValueError(f'Expected input `y_test` to non-empty and 1D `pd.DataFrame`')
    if not isinstance(y_test, pd.Series):
        raise TypeError(f"Expected input {y_test} to be of type 'pd.Series', but received {type(y_test).__name__} instead.")
    if not isinstance(prediction, np.ndarray):
        raise TypeError(f"Expected input {prediction} to be of type 'np.ndarry', but received {type(prediction).__name__} instead.")
    if not isinstance(alpha,float):
        raise TypeError
    if not isinstance(figsize, tuple):
        raise TypeError(f" Expected input {figsize} to be of type 'tuple', but received {type(figsize).__name__} instead.")

    residuals = y_test - prediction

    plt.figure(figsize=figsize)
    plt.scatter(prediction, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel(x_label,fontsize=18)
    plt.ylabel(y_label,fontsize=18)
    plt.title(chart_title,fontsize=chart_title_size)
    plt.yticks(fontsize=xtick_size)
    plt.xticks(fontsize=ytick_size)
    plt.grid(show_grid)
    plt.tight_layout()
    plt.savefig(output_filepath / save_filename, dpi=300)
    plt.show()
    plt.close()
