User Guide
==========

This guide will help you get started with the photon flux estimation package.

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   examples

Installation
-----------
The package can be installed using pip:

.. code-block:: bash

   pip install photon-flux-estimation

Dependencies will be automatically installed.

Quick Start
----------
Here's a simple example of how to use the package:

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

Examples
--------
Check out the examples section for more detailed usage scenarios and best practices.
