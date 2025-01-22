"""Tests for core functionality."""

import numpy as np
import pytest
from ..core import PhotonFluxEstimator


class TestPhotonFluxEstimator:
    """Test suite for PhotonFluxEstimator class."""
    
    @pytest.fixture
    def synthetic_movie(self):
        """Create synthetic movie with known properties."""
        height, width, time = 50, 50, 100
        true_sensitivity = 2.0
        true_zero_level = 100
        
        # Create known photon flux
        photon_flux = np.random.poisson(lam=10, size=(height, width, time))
        
        # Convert to intensity movie
        movie = true_zero_level + true_sensitivity * photon_flux
        
        return movie, true_sensitivity, true_zero_level, photon_flux
    
    def test_initialization(self, synthetic_movie):
        """Test estimator initialization."""
        movie, _, _, _ = synthetic_movie
        estimator = PhotonFluxEstimator(movie)
        
        assert estimator.movie is movie
        assert estimator.sensitivity is None
        assert estimator.zero_level is None
        assert estimator.results is None
        assert estimator.photon_flux is None
    
    def test_initialization_validation(self):
        """Test input validation during initialization."""
        with pytest.raises(AssertionError):
            PhotonFluxEstimator(np.zeros((10, 10)))  # 2D array
        
        with pytest.raises(AssertionError):
            PhotonFluxEstimator(np.zeros((10,)))  # 1D array
    
    def test_compute_sensitivity(self, synthetic_movie):
        """Test sensitivity computation."""
        movie, true_sensitivity, true_zero_level, _ = synthetic_movie
        estimator = PhotonFluxEstimator(movie)
        
        results = estimator.compute_sensitivity()
        
        # Check results are reasonable (within 20% of true values)
        assert np.abs(results['sensitivity'] - true_sensitivity) < 0.4
        assert np.abs(results['zero_level'] - true_zero_level) < 20
        
        # Check all required keys are present
        expected_keys = {'model', 'counts', 'min_intensity', 'max_intensity', 
                        'variance', 'sensitivity', 'zero_level'}
        assert all(key in results for key in expected_keys)
        
        # Check attributes are set
        assert estimator.sensitivity == results['sensitivity']
        assert estimator.zero_level == results['zero_level']
        assert estimator.results is results
    
    def test_compute_sensitivity_insufficient_range(self):
        """Test sensitivity computation with insufficient intensity range."""
        movie = np.ones((50, 50, 100)) * 100
        movie = movie + np.random.normal(0, 0.1, movie.shape)
        estimator = PhotonFluxEstimator(movie)
        
        with pytest.raises(AssertionError) as excinfo:
            estimator.compute_sensitivity()
        assert "sufficient range of intensities" in str(excinfo.value)
    
    def test_compute_photon_flux(self, synthetic_movie):
        """Test photon flux computation."""
        movie, true_sensitivity, true_zero_level, true_photon_flux = synthetic_movie
        estimator = PhotonFluxEstimator(movie)
        
        # Test that error is raised if sensitivity not computed
        with pytest.raises(ValueError):
            estimator.compute_photon_flux()
        
        # Compute sensitivity first
        estimator.compute_sensitivity()
        
        # Now compute photon flux
        computed_flux = estimator.compute_photon_flux()
        
        # Check shape
        assert computed_flux.shape == movie.shape
        
        # Check values are close to true photon flux
        # Note: We use a high tolerance because Poisson noise makes exact matching difficult
        assert np.allclose(computed_flux.mean(), true_photon_flux.mean(), rtol=0.1)
        
        # Check dtype
        assert computed_flux.dtype == np.float32
        
        # Check attribute is set
        assert estimator.photon_flux is computed_flux
    
    def test_plot_analysis(self, synthetic_movie):
        """Test analysis plotting."""
        movie, _, _, _ = synthetic_movie
        estimator = PhotonFluxEstimator(movie)
        
        # Test that error is raised if sensitivity not computed
        with pytest.raises(ValueError):
            estimator.plot_analysis()
        
        # Compute sensitivity
        estimator.compute_sensitivity()
        
        # Test plotting
        fig = estimator.plot_analysis()
        assert len(fig.axes) == 8  # 4 main axes + 4 colorbars
        
        # Test with title
        title = "Test Analysis"
        fig = estimator.plot_analysis(title=title)
        assert fig._suptitle.get_text().startswith(title)
