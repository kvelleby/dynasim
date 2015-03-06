import numpy as np
import pandas as pd

def setup_data(df, start, end):
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
    return(df)
