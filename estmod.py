import statsmodels.api as sm
import patsy
import pandas as pd
import numpy as np
from numpy import log

def est_model(formula, modtype, df, timevar, start):
    '''
    Estimates a regression model given model type,
    formula and df. The function only estimates on
    data from timevar up until start (of simulation).
    '''
    if modtype == 'logit':
        y, X = patsy.dmatrices(formula, 
                df.loc[df.index.get_level_values(timevar) <= start])
        model = sm.Logit(y, X).fit()
    elif modtype == 'identity':
        y, X = patsy.dmatrices(formula, 
                df.loc[df.index.get_level_values(timevar) <= start])
        model = sm.OLS(y, X).fit()
    elif modtype == 'mlogit':
        y, X = patsy.dmatrices(formula,
                df.loc[df.index.get_level_values(timevar) <= start], 
                return_type='dataframe')
        model = sm.MNLogit(y, X).fit()
    else:
        raise TypeError('Unknown model type, %s.' % modtype)
    return(model)

def evaluate_model_call(models, df, modtypes, formulas, timevar, start):
    if models==None:
        models = []
        for modtype, formula in zip(modtypes, formulas):
            model = est_model(formula, modtype, df, timevar, start)
            models.append(model)
    else:
        if len(models) != len(formulas):
            message = 'Length of model list is not equal to formula list'
            raise ValueError, message
        for num, model in enumerate(models):
            if model== None:
                model = est_model(formulas[num], 
                                  modtypes[num], 
                                  df, 
                                  timevar, 
                                  start)
                models.insert(num, model)
    return(models)

