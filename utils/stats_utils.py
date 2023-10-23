import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan, linear_reset
from statsmodels.stats.stattools import jarque_bera


def get_ols_assumption_report(y: pd.DataFrame,
                              X: pd.DataFrame) -> pd.DataFrame:
    """
    Generate an OLS assumption report for each stock in the given DataFrame.

    Parameters:
    - y (pd.DataFrame): The DataFrame containing dependent variable(s) for each stock.
    - X (pd.DataFrame): The DataFrame containing independent variable(s) for each stock.

    Returns:
    - df_report (pd.DataFrame): A DataFrame containing OLS assumption reports for each stock.

    This function performs Ordinary Least Squares (OLS) regression for each stock separately
    and checks several OLS assumptions, including:
    - Collinearity: Checks for perfect collinearity (condition number > 30).
    - Zero Conditional Mean: Assumes if intercept is in the model.
    - Homoscedasticity: Uses the Breusch-Pagan test.
    - Autocorrelation: Uses the Breusch-Godfrey test.
    - Normality: Uses the Jarque-Bera test.
    - Functional Form: Uses the Ramsey RESET test.

    The function returns a DataFrame with assumption reports for each stock.

    Example usage:
    assumption_reports = get_ols_assumption_report(y_data, X_data)
    
    """

    all_reports = {}
        
    # Loop through each stock (assuming columns represent stocks)
    for stock_id in y.columns:
        
        # Isolate the y (dependent variable) for the stock
        y = y[stock_id]
        
        X = sm.add_constant(X[stock_id])
        
        # Fit the OLS model
        model = sm.OLS(y, X).fit()
        
        # Initialize report for this stock
        report = {}
        
        # No Perfect Collinearity
        condition_number = np.linalg.cond(model.model.exog)
        report['Collinearity'] = 'Violated' if condition_number > 30 else 'Not Violated'
        
        # Zero Conditional Mean
        report['Zero Conditional Mean'] = 'Assumed' if 'const' in model.model.exog_names else 'Violated'
        
        # Homoscedasticity: Breusch-Pagan Test
        _, pval, __, f_pval = het_breuschpagan(model.resid, model.model.exog)
        report['Homoscedasticity'] = 'Violated' if pval < 0.05 else 'Not Violated'
        
        # Autocorrelation: Breusch-Godfrey Test
        _, pval, __, f_pval = acorr_breusch_godfrey(model)
        report['No Autocorrelation'] = 'Violated' if pval < 0.05 else 'Not Violated'
        
        # Normality: Jarque-Bera Test
        _, pval, __, __ = jarque_bera(model.resid)
        report['Normality'] = 'Violated' if pval < 0.05 else 'Not Violated'
        
        # Functional Form: Ramsey RESET Test
        reset_test_result = linear_reset(model, power=2)
        pval = reset_test_result.pvalue
        report['Functional Form'] = 'Violated' if pval < 0.05 else 'Not Violated'
        
        # Add the report for this stock to all_reports
        all_reports[stock_id] = report

    df_report = pd.DataFrame(all_reports)

    return df_report
