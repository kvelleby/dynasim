from __future__ import print_function
import pandas as pd
import pdb

# DynaSim modules
from utilities import draw_betas, apply_ts
from simmod import simulate
from estmod import evaluate_model_call
from setupdata import setup_data

# TODO: Support for mlogit
# TODO: Support for hierarchical models

def Simulator(formulas,
            modtypes,
            tslist,
            df,
            groupvar,
            timevar,
            nsim,
            start,
            end,
            models=None):
    '''
    Main program call
    '''
    df = df.reset_index()
    df = df.set_index([timevar, groupvar])
    # TODO: Difference t - (t-1)
    df = apply_ts(df, tslist) 
    models = evaluate_model_call(models, df, modtypes, formulas, timevar, start)
    # TODO: Functionality for choosing own beta-estimates.
    betasli = [draw_betas(model, nsim) for model in models]
    # TODO: Test and report missing data in setup_data
    df = setup_data(df, start, end)
    # TODO: Faster simulation in simulate
    results, summaryvars = simulate(formulas, 
                       betasli, 
                       modtypes, 
                       df, 
                       nsim, 
                       timevar, 
                       start, 
                       end,
                       tslist)
    return(models, betasli, df, results, summaryvars)

