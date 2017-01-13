"""DynaSIM spatial module

This module contains functions for the handling of spatial data. Currently in
development, does not work.
"""

import pysal
import numpy as np
import fiona
import tempfile
import pickle

def subset_shp(shapefile, newfile, filter_fn, t):
    with fiona.open(shapefile) as shp:
        with fiona.open(newfile, 'w', **shp.meta) as new:
            [new.write(rec) for rec in shp if filter_fn(rec, t)]

def filter_cshapes_year(rec, t):
    if rec['properties']['GWSYEAR'] <= t <= rec['properties']['GWEYEAR']:
        return True

def weight_from_shapefile(shapefile, srule='rook', rowstandardize=False):
    '''
    Utility function to get a weight-matrix from any given cross-section
    in cshapes. Should also work as a good example for how to create
    weight-matrices in Python.
    '''
    if srule == 'rook':
        w = pysal.rook_from_shapefile(shapefile)
    elif srule == 'queen':
        w = pysal.queen_from_shapefile(shapefile)
    elif srule == 'bishop':
        wq = pysal.queen_from_shapefile(shapefile)
        wr = pysal.rook_from_shapefile(shapefile)
        w = pysal.w_difference(wq, wr, constrained = False)
    if rowstandardize:
        w.transform = 'r'
    return w

def update_df(df, t, sdict):
    df.loc[t, sdict['name']] = pysal.lag_spatial(sdict['w'],
            df.loc[t, sdict['var']].values)

def apply_spatial_lag(df, sdict, groupvar, timevar, cshapes=False):
    """
    sdict['name']: name of new spatial lag variable
    sdict['w']: spatial weight to apply
    sdict['var']: observed variable to calculate spatial lag of
    """
    if cshapes:
        wq = pickle.load(open('dynasim/data/cshapes_rook.p', 'rb'))
        start = df.index.levels[0].min()
        end = df.index.levels[0].max()

        for t in range(start, end):
            criteria = df.index.get_level_values(timevar) == t
            df.loc[criteria, sdict['name']] = pysal.lag_spatial(wq[t-1946],
                      df.loc[criteria, sdict['var']].values)

    df[sdict['name']] = pysal.lag_spatial(sdict['w'],
            df[sdict['var']].values)
