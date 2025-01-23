"""Tests for visualization functionality."""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from src.photon_flux_estimation.visualization import (
    plot_average_intensity,
    plot_photon_transfer_curve,
    plot_coefficient_variation,
    plot_photon_flux,
)
from src.photon_flux_estimation.core import PhotonFluxEstimator


@pytest.fixture
def test_movie_and_results():
    """Load test movie and compute sensitivity results for testing."""
    # Load test data
    data = np.load("tests/testdata/movie1.npz")
    movie = data["scan"]

    # Compute sensitivity using PhotonFluxEstimator
    estimator = PhotonFluxEstimator(movie)
    results = estimator.compute_sensitivity()

    return movie, results


def test_plot_average_intensity(test_movie_and_results):
    """Test plot_average_intensity function."""
    movie, _ = test_movie_and_results

    # Test basic plotting
    fig = plot_average_intensity(movie)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2  # Main axis + colorbar

    # Test with title
    title = "Test Plot"
    fig = plot_average_intensity(movie, title=title)
    assert fig._suptitle.get_text() == title

    # Test with invalid dimensions
    with pytest.raises(TypeError):
        plot_average_intensity(movie[0])  # Wrong dimensions

    plt.close("all")


def test_plot_photon_transfer_curve(test_movie_and_results):
    """Test plot_photon_transfer_curve function."""
    _, results = test_movie_and_results

    # Test basic plotting
    fig = plot_photon_transfer_curve(results)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1

    # Check axis labels
    ax = fig.axes[0]
    assert ax.get_xlabel().lower() == "intensity"
    assert ax.get_ylabel().lower() == "variance"

    plt.close("all")


def test_plot_coefficient_variation(test_movie_and_results):
    """Test plot_coefficient_variation function."""
    movie, results = test_movie_and_results

    # Test basic plotting
    fig = plot_coefficient_variation(movie, results)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2  # Main axis + colorbar

    # Test with invalid inputs
    with pytest.raises(KeyError):
        plot_coefficient_variation(movie, {})  # Empty results dict

    plt.close("all")


def test_plot_photon_flux(test_movie_and_results):
    """Test plot_photon_flux function."""
    movie, results = test_movie_and_results

    # Test basic plotting
    fig = plot_photon_flux(movie, results)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2  # Main axis + colorbar

    # Test with invalid inputs
    with pytest.raises(KeyError):
        plot_photon_flux(movie, {})  # Empty results dict

    plt.close("all")
