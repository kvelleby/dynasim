import numpy as np
import pandas as pd

def decide(nature, risk): 
    if nature < risk: 
        return 1
    else:
        return 0
# Vectorize the function for use with numpy arrays
vdecide = np.vectorize(decide)

def inv_logit(x):
    return (1 + np.tanh(x / 2)) / 2

def draw_betas(model, modeltype, nsim):
    # Due to a bug in statsmodels.MNLogit (to do with the pandas wrapper),
    # we must use model_results.cov_params as of now.
    if modeltype == 'mlogit':
        return(np.random.multivariate_normal(np.ravel(model.params, order=True),
                                             model._results.cov_params(), 
                                             nsim))
    else:
        return(np.random.multivariate_normal(model.params, 
                                             model.cov_params(), 
                                             nsim))

def find_lhsvars(formulas):
    return([formula.split("~")[0].strip() for formula in formulas])

def apply_ts(df, tslist):
    '''
    Calculates new variables from time-series information.
    '''

    def time_since(var, criteria, tsstart):
        '''
        Count time as long as variable meets criteria, then resets
        tsstart: Starting count value for time-series that enter data 
        '''
        def rolling_count(val, criteria):
            if eval('val'+criteria):
                rolling_count.count +=1
            else:
                rolling_count.count = 1
            return rolling_count.count
        rolling_count.count = tsstart 
        return(var.apply(rolling_count, criteria=criteria))

    for d in tslist:
        if 'lag' in d.keys():
            df[d['name']] = (df.groupby(level=1)[d['var']]
                               .shift(d['lag'])
                            )
            if 'value' in d.keys():
                df[d['name']] = df[d['name']] == d['value']
        if 'ts' in d.keys():
            df[d['name']] = (df.groupby(level=1)[d['var']]
                               .transform(time_since, 
                                          criteria=d['ts'], 
                                          tsstart=d['tsstart'])
                            )
        if 'rmean' in d.keys():
            df[d['name']] = (df.groupby(level=1)[d['var']]
                               .transform(pd.rolling_mean, 
                                          window=d['rmean'])
                            )
    return(df)

