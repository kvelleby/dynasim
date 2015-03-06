# DynaSim 

Dynamic simulation of regression models

DynaSim is a project written in Python 2.7 where the goal is to make a user-friendly interface for simulating dynamic panel-data regression models. It will support all GLM models as well as hierarchical versions and multinomial models. It will also be possible to simulate an arbitrary number of models at each time-point in sequence.

At the moment, the simplest way to use DynaSim is to import it into an IPython session, load your panel-data as a Pandas data-frame, and run dynasim.app.Simulator.

TODO:
- Licence
- Add tests
- Add logging
- Add example
- Missing functionality:
  - More link-functions
  - Hierarchical models
  - Multinomial models
  - Growth-functions in apply_ts()
  - Save results to file
