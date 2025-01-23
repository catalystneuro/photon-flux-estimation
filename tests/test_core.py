"""Tests for core functionality."""

import numpy as np
import pytest
from src.photon_flux_estimation.core import PhotonFluxEstimator


class TestPhotonFluxEstimator:
    """Test suite for PhotonFluxEstimator class."""
    
    @pytest.fixture
    def test_movie(self):
        """Load test movie for testing."""
        # Load test data
        data = np.load('tests/testdata/movie1.npz')
        movie = data['scan']
        
        return movie
    
    def test_initialization(self, test_movie):
        """Test estimator initialization."""
        movie = test_movie
        count_weight_gamma = 1
        estimator = PhotonFluxEstimator(movie=movie,count_weight_gamma=count_weight_gamma)
        
        assert estimator.movie is movie
        assert estimator.count_weight_gamma is count_weight_gamma
        assert estimator.sensitivity is None
        assert estimator.zero_level is None
        assert estimator.results is None
    
    def test_initialization_validation(self):
        """Test input validation during initialization."""
        with pytest.raises(AssertionError):
            PhotonFluxEstimator(np.zeros((10, 10)))  # 2D array
        
        with pytest.raises(AssertionError):
            PhotonFluxEstimator(np.zeros((10,)))  # 1D array
    
    def test_compute_sensitivity(self, test_movie):
        """Test sensitivity computation."""
        movie = test_movie
        estimator = PhotonFluxEstimator(movie)
        
        results = estimator.compute_sensitivity()
                
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
    
    def test_compute_photon_flux(self, test_movie):
        """Test photon flux computation."""
        movie = test_movie
        estimator = PhotonFluxEstimator(movie)
        
        # Now compute photon flux
        computed_flux = estimator.compute_photon_flux()
        
        # Check shape
        assert computed_flux.shape == movie.shape
    
    def test_plot_analysis(self, test_movie):
        """Test analysis plotting."""
        movie = test_movie
        estimator = PhotonFluxEstimator(movie)

        # Test with title
        title = "Test Analysis"
        fig = estimator.plot_analysis(title=title)
        assert fig._suptitle.get_text().startswith(title)
