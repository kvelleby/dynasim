import numpy as np
import pandas as pd
import patsy
import itertools
import re

# Select relevant columns and remove missing:
def find_innerterm(term):
    parenthesis = re.compile("\(")
    if parenthesis.search(term) == None:
        innerterm = term
    else:
        reinner = re.compile("\((.*)\)")
        innerterm = reinner.search(term).group(1)
    return(innerterm)
def find_innerterms(formula):
    env = patsy.EvalEnvironment.capture()
    md = patsy.ModelDesc.from_formula(formula, env)
    terms = list(itertools.chain.from_iterable([md.rhs_termlist, md.lhs_termlist]))
    reterm = re.compile("'([\w\.()]*)'")
    res = [reterm.findall(str(term)) for term in terms]
    flat_res = list(itertools.chain.from_iterable(res))
    innerterms = [find_innerterm(term) for term in flat_res]
    return(innerterms)
def find_dependent(formula):
    return(formula.split('~')[0].strip())
def find_endogvars(formulas, tsvars, spatialdicts):
    depnames = [find_dependent(formula) for formula in formulas]
    tsnames = [d['name'] for d in tsvars]
    spatnames = [d['name'] for d in spatialdicts]
    endogvars = set(depnames + tsnames + spatnames)
    return endogvars

def define_varsets(formulas, tsvars, spatialdicts, groupvar, timevar):
    innerterms = [find_innerterms(formula) for formula in formulas]
    flat_innerterms = list(itertools.chain.from_iterable(innerterms))
    endogset = find_endogvars(formulas, tsvars, spatialdicts)
    innertermset = set(flat_innerterms)
    structvarset = set([groupvar, timevar])
    exogset = innertermset - endogset 
    depset = set([find_dependent(formula) for formula in formulas])
    return endogset, exogset, depset, innertermset, structvarset

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

def apply_ts(df, tsvars):
    '''
    Calculates new variables from time-series information.
    '''

    def count_while(var, criteria):
        '''
        Count time as long as variable meets criteria, then resets
        '''
        def rolling_count(val, criteria):
            if eval('val'+criteria):
                rolling_count.count +=1
            else:
                rolling_count.count = 1
            return rolling_count.count
        rolling_count.count = 0
        return(var.apply(rolling_count, criteria=criteria))

    for d in tsvars:
        if 'lag' in d.keys():
            df[d['name']] = (df.groupby(level=1)[d['var']]
                               .shift(d['lag'])
                            )
            if 'value' in d.keys():
                df[d['name']] = df[d['name']] == d['value']
        if 'cw' in d.keys():
            df[d['name']] = (df.groupby(level=1)[d['var']]
                               .transform(count_while, 
                                          criteria=d['cw'])
                            )
        if 'ma' in d.keys():
            df[d['name']] = (df.groupby(level=1)[d['var']]
                               .transform(pd.rolling_mean, 
                                          window=d['ma'])
                            )
    return(df)

