"""DynaSIM: Main application

This module defines the dynasim class, which is where you contain the
simulation setup, data, and functions.
"""

from __future__ import print_function
from __future__ import division
import pandas as pd
import pdb

# DynaSim modules
from dynasim.utilities import draw_betas, apply_ts, define_varsets
from dynasim.spatial import apply_spatial_lag
from dynasim.simmod import simulate
from dynasim.estmod import evaluate_model_call
from dynasim.setupdata import setup_data

# TODO: Support for hierarchical models
class DynaSim(object):

    """Docstring for . """

    def __init__(self,
                 formulas,
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

        self.formulas = formulas
        self.modtypes = modtypes
        self.tsvars = tsvars
        self.spatialdicts = spatialdicts
        self.groupvar = groupvar
        self.timevar = timevar
        self.nsim = nsim
        self.start = start
        self.end = end
        self.filename = filename
        self.models = models

        self.df = df
        self.df = self.df.reset_index()
        self.df = self.df.set_index([timevar, groupvar], drop=False)

        self.varsets = define_varsets(self.formulas, self.tsvars, self.spatialdicts, self.groupvar, self.timevar)

    def calculate_model_ts(self):
        self.df = apply_ts(self.df, self.tsvars)
    def calculate_model_spatial_vars(self, cshapes=True):
        [apply_spatial_lag(self.df, sdict, self.groupvar, self.timevar, cshapes)
                for sdict in self.spatialdicts]

    def estimate(self):
        self.models = evaluate_model_call(self.models,
                                          self.df,
                                          self.modtypes,
                                          self.formulas,
                                          self.timevar,
                                          self.start)
    def calculate_betas(self):
        self.betasli = [draw_betas(model, modtype, self.nsim)
                        for model, modtype in zip(self.models, self.modtypes)]

    def setup_simulation_data(self):
        self.simdf = setup_data(self.df,
                                self.varsets['innertermset'],
                                self.varsets['exogset'],
                                self.timevar,
                                self.groupvar,
                                self.start,
                                self.end)

    def sim(self):
        self.results, self.summaryvars = simulate(self.formulas,
                                                  self.betasli,
                                                  self.modtypes,
                                                  self.models,
                                                  self.simdf,
                                                  self.nsim,
                                                  self.timevar,
                                                  self.start,
                                                  self.end,
                                                  self.tsvars,
                                                  self.spatialdicts,
                                                  self.filename)
