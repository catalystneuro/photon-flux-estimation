"""Tests for visualization functionality."""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from ..visualization import (
    plot_average_intensity,
    plot_photon_transfer_curve,
    plot_coefficient_variation,
    plot_photon_flux,
    plot_sensitivity_analysis
)
from ..core import compute_sensitivity


@pytest.fixture
def synthetic_movie_and_results():
    """Create synthetic movie and compute sensitivity results for testing."""
    # Create synthetic movie
    height, width, time = 50, 50, 100
    true_sensitivity = 2.0
    true_zero_level = 100
    
    # Generate synthetic data
    photon_flux = np.random.poisson(lam=10, size=(time, height, width))
    movie = true_zero_level + true_sensitivity * photon_flux
    movie = movie.transpose(1, 2, 0)  # Convert to (height, width, time)
    movie = movie + np.random.normal(0, 1, movie.shape)
    
    # Compute sensitivity
    results = compute_sensitivity(movie)
    
    return movie, results


def test_plot_average_intensity(synthetic_movie_and_results):
    """Test plot_average_intensity function."""
    movie, _ = synthetic_movie_and_results
    
    # Test basic plotting
    fig = plot_average_intensity(movie)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2  # Main axis + colorbar
    
    # Test with title
    title = "Test Plot"
    fig = plot_average_intensity(movie, title=title)
    assert fig._suptitle.get_text() == title
    
    # Test saving (using tmp_path fixture)
    with pytest.raises(TypeError):
        plot_average_intensity(movie[0])  # Wrong dimensions
    
    plt.close('all')


def test_plot_photon_transfer_curve(synthetic_movie_and_results):
    """Test plot_photon_transfer_curve function."""
    _, results = synthetic_movie_and_results
    
    # Test basic plotting
    fig = plot_photon_transfer_curve(results)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    
    # Check axis labels
    ax = fig.axes[0]
    assert ax.get_xlabel().lower() == 'intensity'
    assert ax.get_ylabel().lower() == 'variance'
    
    plt.close('all')


def test_plot_coefficient_variation(synthetic_movie_and_results):
    """Test plot_coefficient_variation function."""
    movie, results = synthetic_movie_and_results
    
    # Test basic plotting
    fig = plot_coefficient_variation(movie, results)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2  # Main axis + colorbar
    
    # Test with invalid inputs
    with pytest.raises(KeyError):
        plot_coefficient_variation(movie, {})  # Empty results dict
    
    plt.close('all')


def test_plot_photon_flux(synthetic_movie_and_results):
    """Test plot_photon_flux function."""
    movie, results = synthetic_movie_and_results
    
    # Test basic plotting
    fig = plot_photon_flux(movie, results)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2  # Main axis + colorbar
    
    # Test with invalid inputs
    with pytest.raises(KeyError):
        plot_photon_flux(movie, {})  # Empty results dict
    
    plt.close('all')


def test_plot_sensitivity_analysis(synthetic_movie_and_results):
    """Test plot_sensitivity_analysis function."""
    movie, results = synthetic_movie_and_results
    
    # Test basic plotting
    fig = plot_sensitivity_analysis(movie, results)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 8  # 4 main axes + 4 colorbars
    
    # Test with title
    title = "Test Analysis"
    fig = plot_sensitivity_analysis(movie, results, title=title)
    assert fig._suptitle.get_text().startswith(title)
    
    # Test saving (using tmp_path fixture)
    with pytest.raises(TypeError):
        plot_sensitivity_analysis(movie[0], results)  # Wrong dimensions
    
    plt.close('all')
