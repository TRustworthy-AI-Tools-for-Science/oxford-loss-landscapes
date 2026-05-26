"""
Oxford Loss Landscapes: A library for visualizing and analyzing neural network loss landscapes

This package provides tools for:
- Computing and visualizing loss landscapes of neural networks
- Analyzing Hessian properties of loss functions
- Downloading and managing models
- Dashboard utilities for data analysis

Installation:
    Simple one-command installation with all dependencies:
    
    $ pip install -e .
    
    This automatically installs PyTorch, NumPy, and all other dependencies
    with correct versions for your Python version.

Example:
    Basic usage for computing a loss landscape:

    >>> import oxford_loss_landscapes as oll
    >>> from oxford_loss_landscapes import ModelWrapper
    >>> 
    >>> # Wrap your model
    >>> model_wrapper = ModelWrapper(model, criterion, inputs, targets)
    >>> 
    >>> # Compute loss landscape
    >>> landscape = oll.random_plane(model_wrapper, distance=1.0, steps=51)
"""

import logging

try:
    from ._version import __version__
except ImportError:
    # Package not installed, use fallback version
    __version__ = "0.3.0"

__author__ = "Oxford RSE Project Contributors"
__email__ = "your.email@example.com"

# Main functionality
from .main import (
    point,
    linear_interpolation,
    random_line,
    planar_interpolation,
    random_plane,
    hessian_plane,
)

# Model interface
from .model_interface.model_wrapper import ModelWrapper, GeneralModelWrapper, SimpleModelWrapper, TransformerModelWrapper
from .model_interface.model_parameters import rand_u_like

# Metrics
from .metrics import *

# Contrib modules
from .contrib import *

# Dashboard (optional import - requires advanced dependencies)
try:
    import streamlit
    import plotly
    from .dashboard import *
except ImportError as e:
    # Dashboard dependencies (streamlit, plotly) not available
    # Install with: pip install -e ".[advanced]"
    logging.debug("Optional dashboard dependencies not available: %s", e)
    logging.debug("For dashboard functionality, please install: pip install -e '.[advanced]'")

# Hessian utilities — VR-PCA eigensolver
from .hessian.vrpca import (
    VRPCAConfig,
    VRPCAResult,
    top_hessian_eigenpair_vrpca,
    min_hessian_eigenpair_vrpca,
    min_max_hessian_eigs_vrpca,
)

# Utilities
from .download_models import download_zenodo_model, download_zenodo_zip, extract_zip

__all__ = [
    "point",
    "linear_interpolation",
    "random_line",
    "planar_interpolation",
    "random_plane",
    "hessian_plane",
    "ModelWrapper",
    "GeneralModelWrapper",
    "SimpleModelWrapper",
    "TransformerModelWrapper",
    "rand_u_like",
    "VRPCAConfig",
    "VRPCAResult",
    "top_hessian_eigenpair_vrpca",
    "min_hessian_eigenpair_vrpca",
    "min_max_hessian_eigs_vrpca",
    "download_zenodo_model",
    "download_zenodo_zip",
    "extract_zip",
]
