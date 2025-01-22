Photon Flux Estimation Documentation
================================

A Python library for estimating photon flux from two-photon imaging data.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user_guide/index
   api/index

Features
--------
- Compute photon sensitivity from imaging data
- Estimate photon flux
- Generate comprehensive visualizations
- Support for various data formats

Installation
------------
.. code-block:: bash

   pip install photon-flux-estimation

Quick Start
----------
.. code-block:: python

   import numpy as np
   from photon_flux_estimation import PhotonFluxEstimator

   # Load your movie data (height, width, time)
   movie = ...  # Your 3D movie data

   # Create estimator
   estimator = PhotonFluxEstimator(movie)

   # Compute sensitivity
   results = estimator.compute_sensitivity()

   # Compute photon flux
   photon_flux = estimator.compute_photon_flux()

   # Generate visualization
   fig = estimator.plot_analysis()
