import pandas as pd
import numpy as np
from numpy import log
import patsy
from utilities import vdecide, inv_logit, find_lhsvars, apply_ts
import pdb

def simulate(formulas, betasli, modtypes, models, df, nsim, timevar, start, end, tslist):

    def multinom_sim(X, model, b):
        nparam = len(model.params)
        K = (b.shape[0]/nparam) + 1
        b=b.reshape((K-1, nparam))
        nobs = X.shape[0]

        probs = np.zeros((K, nobs))
        for k in range(K-1):
            probs[k+1] = np.exp(X.dot(b[k])) / (1 + np.sum(np.exp(X.dot(b.T)), axis=1))
        probs[0] = 1 - np.sum(probs, axis=0)
        colnames = [(lhsvar+str(num), 'int64') for num, k in enumerate(range(K))]
        outcomes = np.array([tuple(np.random.multinomial(1, prob)) for prob in probs.T], 
            dtype=colnames)
        return(probs.T, outcomes)
    
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
    
    for lhsvar, modtype, model, betas in zip(lhsvars, modtypes, models, betasli):
        if modtype == 'logit':
            name = 'p_'+lhsvar
            df[name] = df[lhsvar]
            summaryvars.append(name)
        if modtype == 'mlogit':
            nparam = len(model.params)
            K = (betas.shape[0]/nparam) + 1
            colnames = ['p_'+lhsvar+str(num) for num, k in enumerate(range(K))]
            for num, name in enumerate(colnames):
                df[name] = (df[lhsvar]==num).astype(int)
                summaryvars.append(name)

    # Find shape of result matrix
    placeholder = np.array(df[summaryvars].loc[timeselection])
    shp = placeholder.shape
    shp = list(shp)
    shp.append(nsim)
    result = np.empty(tuple(shp))
    for sim, t in sim_time_generator(timepoints, sims):
        #print(sim, t)
        df = apply_ts(df, tslist)
        for lhsvar, betas, formula, model, modtype in  zip(
                lhsvars, betasli, formulas, models, modtypes):
            y, X = patsy.dmatrices(formula, df.ix[t])
            b = betas[sim].T
            if modtype == 'identity':
                df.loc[t, lhsvar]  = X.dot(b)
            if modtype == 'logit':
                name = 'p_'+lhsvar
                df.loc[t, name]  = inv_logit(X.dot(b))
                nature = np.random.uniform(size=(nunits))
                df.loc[t, lhsvar]  = vdecide(nature, df.loc[t, name])
            if modtype == 'mlogit':
                # This structure assumes strict naming-conventions
                # 0 is base, then next outcomes must be consequtive 1,2,3..etc.
                probs, outcomes = multinom_sim(X, model, b)
                outcomes = pd.DataFrame(outcomes)
                colnames = list(outcomes.columns)
                uvalues = np.array(range(outcomes.shape[1]))+1
                flat_outcome = np.sum(outcomes*uvalues, axis=1)
                flat_outcome =  np.array(flat_outcome, dtype=np.float64)
                df.loc[t, lhsvar] = flat_outcome
                pnames = ['p_'+name for name in colnames]
                df.loc[t, pnames] = probs
        result[:,:,sim] = np.array(df[summaryvars].loc[timeselection])

    return(result, summaryvars)

