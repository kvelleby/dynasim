import numpy as np
import pandas as pd
import patsy
import re
import itertools
import pdb

def setup_data(df, formulas, timevar, start, end):
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
    innerterms = [find_innerterms(formula) for formula in formulas]
    flat_innerterms = list(itertools.chain.from_iterable(innerterms))
    df = df[flat_innerterms].dropna()

    # Only retain data needed for simulation:
    allunits = np.array(df.index.levels[1])
    alltime = np.array(df.index.levels[0])
    time_to_drop = [t for t in alltime if t > end]

    units_to_drop = set()
    for year in range(start-2, end+1):
        l = df.loc[year].index
        l = [unit for unit in allunits if unit not in l]
        # Union operator |=
        units_to_drop |= set(l)
        
    units_to_drop = list(units_to_drop)

    df = df.to_panel()
    if len(units_to_drop) > 0:
        df.drop(units_to_drop, axis='minor', inplace=True)
    if len(time_to_drop) > 0:
        df.drop(time_to_drop, axis='major', inplace=True)
    df = df.to_frame()
    
    # Test if unit exists for each consecutive time-point
    def is_consecutive(ts):
        not_consecutive = [(a,b) for a,b in zip(ts,ts[1:]) if b != a+1]
        return(len(not_consecutive)==0)
    def test_is_consecutive(df, col):
        df = df.reset_index()
        return(is_consecutive(df[col]))
    def remove_not_consecutive(df, col):
        consecutive_ts = df.groupby(level=1).apply(test_is_consecutive, col)
        not_consecutive_ts = list(consecutive_ts[consecutive_ts == False])
        df = df.to_panel()
        df.drop(not_consecutive_ts, axis='minor', inplace=True)
        df = df.to_frame()
        return(df)
        
    df = remove_not_consecutive(df, timevar)


    return(df)
