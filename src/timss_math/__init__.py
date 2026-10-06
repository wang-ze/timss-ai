"""TIMSS mathematics items: the prediction of their IRT parameters.

- ``item_parameter_prediction``: predicts the IRT parameters of new items from the most similar calibrated items,
  with the released items and their calibrated parameters as package data.

The released-item datasets are built by ``notebooks/scratch/get_data.ipynb``, a local notebook that is not in the
repository, with the helpers in ``src/released_items.py``; ``notebooks/timss_math_released_items.md`` documents how.
"""
