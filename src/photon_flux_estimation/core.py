"""Core functionality for photon flux estimation."""

import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from sklearn.linear_model import HuberRegressor as Regressor


def _longest_run(bool_array: np.ndarray) -> slice:
    """Find the longest contiguous segment of True values inside bool_array."""
    step = np.diff(np.int8(bool_array), prepend=0, append=0)
    on = np.where(step == 1)[0]
    off = np.where(step == -1)[0]
    i = np.argmax(off - on)
    return slice(on[i], off[i])


class PhotonFluxEstimator:
    """Class for estimating photon flux from two-photon imaging data.
    
    This class provides methods for computing photon sensitivity, estimating photon flux,
    and generating various visualizations of the estimation results.
    
    Attributes:
        movie: The input movie data in format (height, width, time)
        sensitivity: Estimated photon sensitivity (gain)
        zero_level: Estimated zero level (offset)
        results: Full dictionary of estimation results
        photon_flux: Computed photon flux movie
    """
    
    def __init__(self, movie: np.ndarray):
        """Initialize with a movie.
        
        Args:
            movie: Input movie data in format (height, width, time)
        """
        assert (
            movie.ndim == 3
        ), f"A three dimensional (Height, Width, Time) grayscale movie is expected, got {movie.ndim}"
        
        self.movie = movie
        self.sensitivity = None
        self.zero_level = None
        self.results = None
        self.photon_flux = None
        
    def compute_sensitivity(self, count_weight_gamma: float = 0.2) -> dict:
        """Calculate photon sensitivity from the movie.
        
        Args:
            count_weight_gamma: Weight parameter for intensity levels (default: 0.2).
                - 0.00001: weigh each intensity level equally
                - 1.0: weigh each intensity in proportion to pixel counts
        
        Returns:
            dict: Estimation results dictionary
        """
        movie = np.maximum(0, self.movie.astype(np.int32, copy=False))
        intensity = (movie[:, :, :-1] + movie[:, :, 1:] + 1) // 2
        difference = movie[:, :, :-1].astype(np.float32) - movie[:, :, 1:]

        select = intensity > 0
        intensity = intensity[select]
        difference = difference[select]

        counts = np.bincount(intensity.flatten())
        bins = _longest_run(counts > 0.01 * counts.mean())
        bins = slice(max(bins.stop * 3 // 100, bins.start), bins.stop)
        assert (
            bins.stop - bins.start > 100
        ), "The image does not have a sufficient range of intensities"

        counts = counts[bins]
        idx = (intensity >= bins.start) & (intensity < bins.stop)
        variance = (
            np.bincount(
                intensity[idx] - bins.start,
                weights=(difference[idx] ** 2) / 2,
            )
            / counts
        )
        
        model = Regressor()
        model.fit(np.c_[bins], variance, counts ** count_weight_gamma)
        
        self.sensitivity = model.coef_[0]
        self.zero_level = -model.intercept_ / model.coef_[0]
        self.results = dict(
            model=model,
            counts=counts,
            min_intensity=bins.start,
            max_intensity=bins.stop,
            variance=variance,
            sensitivity=self.sensitivity,
            zero_level=self.zero_level,
        )
        
        return self.results
    
    def compute_photon_flux(self) -> np.ndarray:
        """Convert raw intensity movie to photon flux movie.
        
        Returns:
            np.ndarray: Photon flux movie in format (height, width, time)
        
        Raises:
            ValueError: If sensitivity hasn't been computed yet
        """
        if self.sensitivity is None:
            raise ValueError("Must compute sensitivity first using compute_sensitivity()")
        
        # Convert to float to avoid integer division
        movie = self.movie.astype(np.float32)
        
        # Compute photon flux by removing the zero level and any spatial gradient
        # We estimate the gradient by taking the mean over time
        mean_frame = np.mean(movie, axis=2, keepdims=True)
        movie_detrended = movie - (mean_frame - np.mean(mean_frame))
        
        # Now compute photon flux using the detrended movie
        self.photon_flux = (movie_detrended - self.zero_level) / self.sensitivity
        
        return self.photon_flux
    
    def plot_analysis(self, title: str = None, save_path: str = None) -> plt.Figure:
        """Generate comprehensive visualization of all estimation results.
        
        Args:
            title: Optional title for the figure
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The generated figure
            
        Raises:
            ValueError: If sensitivity hasn't been computed yet
        """
        if self.results is None:
            raise ValueError("Must compute sensitivity first using compute_sensitivity()")
            
        fig, axx = plt.subplots(2, 2, figsize=(12, 12), tight_layout=True)
        axx = iter(axx.flatten())
        
        # A. Average intensity
        ax = next(axx)
        m = self.movie.mean(axis=0)
        im = ax.imshow(m, vmin=0, vmax=np.quantile(m, 0.999), cmap='gray')
        ax.axis(False)
        plt.colorbar(im, ax=ax)
        ax.set_title('Average Intensity')
        ax.text(-0.1, 1.15, "A", transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')
        
        # B. Photon transfer curve
        ax = next(axx)
        x = np.arange(self.results["min_intensity"], self.results["max_intensity"])
        fit = self.results["model"].predict(x.reshape(-1, 1))
        ax.scatter(x, np.minimum(fit[-1]*2, self.results["variance"]), s=2, alpha=0.5)
        ax.plot(x, fit, 'r')
        ax.grid(True)
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Variance')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title('Photon Transfer Curve')
        ax.text(-0.1, 1.15, "B", transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')
        
        # C. Coefficient of variation
        ax = next(axx)
        v = ((self.movie[1:,:,:].astype('float64') - self.movie[:-1,:,:]) ** 2/2).mean(axis=0)
        imx = np.stack(((m-self.zero_level)/self.sensitivity, 
                       v/self.sensitivity/self.sensitivity, 
                       (m-self.zero_level)/self.sensitivity), axis=-1)
        im = ax.imshow(
            np.minimum(1, np.sqrt(0.01 + np.maximum(0, imx/np.quantile(imx, 0.9999))) - 0.1),
            cmap='PiYG'
        )
        cbar = plt.colorbar(im, ax=ax, ticks=[0.2, .5, 0.8])
        cbar.ax.set_yticklabels(['<< 1', '1', '>> 1'])
        ax.axis(False)
        ax.set_title('Coefficient of Variation')
        ax.text(-0.1, 1.15, "C", transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')
        
        # D. Photon flux
        ax = next(axx)
        if self.photon_flux is None:
            self.compute_photon_flux()
        im = self.photon_flux.mean(axis=0)
        mx = np.quantile(im, 0.999)
        im = ax.imshow(im, vmin=-mx, vmax=mx, cmap=cc.cm.CET_D13)
        plt.colorbar(im, ax=ax)
        ax.axis(False)
        ax.set_title('Photon Flux\n(photons/pixel/frame)')
        ax.text(-0.1, 1.15, "D", transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')
        
        if title:
            plt.suptitle(f'{title}\nPhoton sensitivity: {self.sensitivity:4.1f}')
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
