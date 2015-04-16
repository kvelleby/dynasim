from __future__ import print_function
import pandas as pd
import pdb

# DynaSim modules
from utilities import draw_betas, apply_ts, define_varsets
from spatial import apply_spatial_lag
from simmod import simulate
from estmod import evaluate_model_call
from setupdata import setup_data
import numpy as np

# TODO: Support for hierarchical models

def Simulator(formulas,
            modtypes,
            tsvars,
            spatialdicts,
            df,
            groupvar,
            timevar,
            nsim,
            start,
            end,
            filename='',
            models=None):
    '''
    Main program call
    '''
    df = df.reset_index()
    df = df.set_index([timevar, groupvar], drop=False)
    endogset, exogset, depset, innertermset, structvarset = define_varsets(formulas, 
                                                             tsvars, 
                                                             spatialdicts,
                                                             groupvar,
                                                             timevar)
    # TODO: Difference t - (t-1)
    df = apply_ts(df, tsvars)
    if len(spatialdicts) > 0:
        [apply_spatial_lag(df, sdict, groupvar, timevar, cshapes=True) 
                for sdict in spatialdicts]
    models = evaluate_model_call(models, df, modtypes, formulas, timevar, start)
    # TODO: Functionality for choosing own beta-estimates.
    betasli = [draw_betas(model, modtype, nsim) 
               for model, modtype in zip(models, modtypes)]
    # TODO: Test and report missing data in setup_data
    df = setup_data(df, innertermset, exogset, timevar, groupvar, start, end)
    # TODO: Faster simulation in simulate
    results, summaryvars = simulate(formulas, 
                       betasli, 
                       modtypes, 
                       models,
                       df, 
                       nsim, 
                       timevar, 
                       start, 
                       end,
                       tsvars,
                       spatialdicts,
                       filename)
    return(models, betasli, df, results, summaryvars)

