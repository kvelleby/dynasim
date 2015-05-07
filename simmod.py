import pandas as pd
import numpy as np
from numpy import log
import patsy
import h5py
from utilities import vdecide, inv_logit, find_lhsvars, apply_ts
import streamers
import spatial
import pdb

def simulate(formulas, betasli, modtypes, models, df, nsim, timevar, start, end,
        tsvars, spatialdicts, filename):
    '''
    Documentation
    '''

    def multinom_sim(X, model, beta):
        '''
        Returns two  K*nobs matrices, one for probabilities and one for
        simulated outcomes.
        '''
        nparam = len(model.params)
        K = (beta.shape[0]/nparam) + 1
        beta=beta.reshape((K-1, nparam))
        nobs = X.shape[0]

        probs = np.zeros((K, nobs))
        for k in range(K-1):
            probs[k+1] = np.exp(X.dot(beta[k])) / (1 + np.sum(np.exp(X.dot(beta.T)), axis=1))
        probs[0] = 1 - np.sum(probs, axis=0)
        colnames = [(lhsvar+str(num), 'int64') for num, k in enumerate(range(K))]
        outcomes = np.array([tuple(np.random.multinomial(1, prob)) for prob in probs.T], 
            dtype=colnames)
        return(probs.T, outcomes)
    
    timeselection = np.logical_and(df.index.get_level_values(timevar)>=start,
                     df.index.get_level_values(timevar)<=end)
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
            K = (betas.shape[1]/nparam) + 1
            colnames = ['p_'+lhsvar+str(num) for num, k in enumerate(range(K))]
            for num, name in enumerate(colnames):
                df[name] = (df[lhsvar]==num).astype(int)
                summaryvars.append(name)

    # Find shape of result matrix
    placeholder = np.array(df[summaryvars].loc[timeselection])
    shp = placeholder.shape
    shp = list(shp)
    shp.insert(0,nsim)
    if filename == '':
        result = np.empty(tuple(shp))
    else:
        f = h5py.File(filename, 'w')
        result = f.create_dataset("simulation_results", 
                                  tuple(shp), 
                                  dtype='float64',
                                  compression='lzf')
    # Replace nan with -99 as patsy will remove all observations
    df = df.fillna(-99)
    for sim in range(nsim):
        tsstreams = [streamers.init_order(nunits, tsvar) for tsvar in tsvars]
        for t in range(start+1, end+1):
            for stream in tsstreams:
                update = streamers.tick(
                        stream['streamers'], df.loc[t-1, stream['var']].values)
                df.loc[t, stream['name']] = update
            [spatial.update_df(df, t, sdict) for sdict in spatialdicts]
            for lhsvar, betas, formula, model, modtype in  zip(
                    lhsvars, betasli, formulas, models, modtypes):
                y, X = patsy.dmatrices(formula, df.ix[t])
                beta = betas[sim].T
                #print(sim, t, modtype)
                if modtype == 'identity':
                    outcome = X.dot(beta)
                    df.loc[t, lhsvar]  = outcome
                if modtype == 'logit':
                    name = 'p_'+lhsvar
                    df.loc[t, name]  = inv_logit(X.dot(beta))
                    nature = np.random.uniform(size=(nunits))
                    df.loc[t, lhsvar]  = vdecide(nature, df.loc[t, name])
                if modtype == 'mlogit':
                    # This structure assumes strict naming-conventions
                    # 0 is base, then next outcomes must be consequtive 1,2,3..etc.
                    probs, outcomes = multinom_sim(X, model, beta)
                    outcomes = pd.DataFrame(outcomes)
                    colnames = list(outcomes.columns)
                    uvalues = np.array(range(outcomes.shape[1]))
                    flat_outcome = np.sum(outcomes*uvalues, axis=1)
                    flat_outcome =  np.array(flat_outcome, dtype=np.float64)
                    df.loc[t, lhsvar] = flat_outcome
                    pnames = ['p_'+name for name in colnames]
                    df.loc[t, pnames] = probs

        result[sim,:,:] = np.array(df[summaryvars].loc[timeselection], dtype=np.float64)
    if filename != '':
        f.close()

    return(result, summaryvars)

