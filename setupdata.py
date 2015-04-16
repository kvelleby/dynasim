import numpy as np
import pandas as pd
import pdb

def setup_data(df, innertermset, exogset, timevar, groupvar, start, end):
    # Here, should only drop if exogenous indepvars are missing. 
    innerterms = list(innertermset)
    exogterms = list(exogset)
    df = df[innerterms].dropna(subset=exogterms)

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
    df = df.to_frame(filter_observations=False)
    # Here I must again remove observations that where dropped.
    df = df.dropna(subset=exogterms)
    
    # Test if unit exists for each consecutive time-point
    def is_consecutive(ts):
        not_consecutive = [(a,b) for a,b in zip(ts,ts[1:]) if b != a+1]
        return(len(not_consecutive)==0)
    def test_is_consecutive(df, col):
        df = df.reset_index()
        return(is_consecutive(df[col]))
    def remove_not_consecutive(df, col):
        df2 = df.copy()
        try:
            df2 = df2.drop(groupvar, axis=1)
        except ValueError:
            pass
        try:
            df2 = df2.drop(timevar, axis=1)
        except ValueError:
            pass
        consecutive_ts = df2.groupby(level=1).apply(test_is_consecutive, col)
        not_consecutive_ts = list(consecutive_ts[consecutive_ts == False])
        df = df.to_panel()
        df.drop(not_consecutive_ts, axis='minor', inplace=True)
        df = df.to_frame(filter_observations=False)
        return(df)
        
    df = remove_not_consecutive(df, timevar)

    return(df)
