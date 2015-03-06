import pandas as pd
import numpy as np
from numpy import log
import patsy
from utilities import vdecide, inv_logit, find_lhsvars, apply_ts

def simulate(formulas, betasli, modtypes, df, nsim, timevar, start, end, tslist):
    
    def sim_time_generator(timepoints, sims):
        for sim in sims:
            for t in timepoints:
                yield sim,t

    timeselection = np.logical_and(df.index.get_level_values(timevar)>=start,
                     df.index.get_level_values(timevar)<=end)
    timepoints = range(start+1, end+1)
    sims = range(0, nsim)
    lhsvars = find_lhsvars(formulas)
    summaryvars = lhsvars[:]
    nunits = len(df.loc[start].index)
    
    for lhsvar, modtype in zip(lhsvars, modtypes):
        if modtype == 'logit':
            name = 'p_'+lhsvar
            df[name] = df[lhsvar]
            summaryvars.append(name)

    # Find shape of result matrix
    placeholder = np.array(df[summaryvars].loc[timeselection])
    shp = placeholder.shape
    shp = list(shp)
    shp.append(nsim)
    result = np.empty(tuple(shp))
    for sim, t in sim_time_generator(timepoints, sims):
        df = apply_ts(df, tslist)
        for lhsvar, betas, formula, modtype in zip(lhsvars, betasli, formulas, modtypes):
            y, X = patsy.dmatrices(formula, df.ix[t])
            b = betas[sim].T
            #print(t, formula, link, X.shape)
            #print(df.ix[t])
            if modtype == 'identity':
                df.loc[t, lhsvar]  = X.dot(b)
                #print(np.all(df.loc[t, lhsvar] > 0))
            if modtype == 'logit':
                name = 'p_'+lhsvar
                df.loc[t, name]  = inv_logit(X.dot(b))
                nature = np.random.uniform(size=(nunits))
                df.loc[t, lhsvar]  = vdecide(nature, df.loc[t, name])
        result[:,:,sim] = np.array(df[summaryvars].loc[timeselection])

    return(result, summaryvars)

