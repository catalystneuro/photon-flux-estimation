"""Visualization functions for photon flux estimation results."""

import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc


def _create_figure(title: str = None) -> tuple:
    """Create a figure and axis with common styling.
    
    Args:
        title: Optional title for the figure
        
    Returns:
        tuple: (figure, axis) matplotlib objects
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    if title:
        fig.suptitle(title)
    return fig, ax


def _add_colorbar(ax, im, ticks=None, labels=None):
    """Add a colorbar to the plot with optional custom ticks and labels.
    
    Args:
        ax: matplotlib axis
        im: matplotlib image object
        ticks: Optional list of tick positions
        labels: Optional list of tick labels
    """
    cbar = plt.colorbar(im, ax=ax)
    if ticks is not None and labels is not None:
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels(labels)
    return cbar


def plot_average_intensity(movie: np.ndarray, title: str = None, save_path: str = None):
    """Plot average intensity across all frames.
    
    Args:
        movie: Input movie data in format (height, width, time)
        title: Optional title for the figure
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The generated figure object
    """
    fig, ax = _create_figure(title)
    
    m = movie.mean(axis=0)
    im = ax.imshow(m, vmin=0, vmax=np.quantile(m, 0.999), cmap='gray')
    ax.axis(False)
    _add_colorbar(ax, im)
    ax.set_title('Average Intensity')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_photon_transfer_curve(results: dict, title: str = None, save_path: str = None):
    """Plot photon transfer curve showing intensity vs variance relationship.
    
    Args:
        results: Dictionary of results from compute_sensitivity()
        title: Optional title for the figure
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The generated figure object
    """
    fig, ax = _create_figure(title)
    
    x = np.arange(results["min_intensity"], results["max_intensity"])
    fit = results["model"].predict(x.reshape(-1, 1))
    ax.scatter(x, np.minimum(fit[-1]*2, results["variance"]), s=2, alpha=0.5)
    ax.plot(x, fit, 'r')
    ax.grid(True)
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Variance')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Photon Transfer Curve')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_coefficient_variation(movie: np.ndarray, results: dict, title: str = None, save_path: str = None):
    """Plot coefficient of variation visualization.
    
    Args:
        movie: Input movie data in format (height, width, time)
        results: Dictionary of results from compute_sensitivity()
        title: Optional title for the figure
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The generated figure object
    """
    fig, ax = _create_figure(title)
    
    q = results['sensitivity']
    b = results['zero_level']
    m = movie.mean(axis=0)
    v = ((movie[1:,:,:].astype('float64') - movie[:-1,:,:]) ** 2/2).mean(axis=0)
    imx = np.stack(((m-b)/q, v/q/q, (m-b)/q), axis=-1)
    im = ax.imshow(
        np.minimum(1, np.sqrt(0.01 + np.maximum(0, imx/np.quantile(imx, 0.9999))) - 0.1),
        cmap='PiYG'
    )
    _add_colorbar(ax, im, ticks=[0.2, .5, 0.8], labels=['<< 1', '1', '>> 1'])
    ax.axis(False)
    ax.set_title('Coefficient of Variation')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_photon_flux(movie: np.ndarray, results: dict, title: str = None, save_path: str = None):
    """Plot photon flux estimation.
    
    Args:
        movie: Input movie data in format (height, width, time)
        results: Dictionary of results from compute_sensitivity()
        title: Optional title for the figure
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The generated figure object
    """
    fig, ax = _create_figure(title)
    
    im = (movie.mean(axis=0)-results['zero_level'])/results['sensitivity']
    mx = np.quantile(im, 0.999)
    im = ax.imshow(im, vmin=-mx, vmax=mx, cmap=cc.cm.CET_D13)
    _add_colorbar(ax, im)
    ax.axis(False)
    ax.set_title('Photon Flux\n(photons/pixel/frame)')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_sensitivity_analysis(movie: np.ndarray, results: dict, title: str = None, save_path: str = None):
    """Generate a comprehensive visualization combining all analysis plots.
    
    Creates a figure with four subplots:
    A. Average intensity image
    B. Photon transfer curve
    C. Coefficient of variation
    D. Photon flux estimation
    
    Args:
        movie: Input movie data in format (height, width, time)
        results: Dictionary of results from compute_sensitivity()
        title: Optional title for the figure
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The generated figure object
        
    Notes:
        This implementation is based on the compress-multiphoton package:
        https://github.com/datajoint/compress-multiphoton
    """
    fig, axx = plt.subplots(2, 2, figsize=(12, 12), tight_layout=True)
    axx = iter(axx.flatten())
    
    # A. Average intensity
    ax = next(axx)
    m = movie.mean(axis=0)
    im = ax.imshow(m, vmin=0, vmax=np.quantile(m, 0.999), cmap='gray')
    ax.axis(False)
    _add_colorbar(ax, im)
    ax.set_title('Average Intensity')
    ax.text(-0.1, 1.15, "A", transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='right')
    
    # B. Photon transfer curve
    ax = next(axx)
    x = np.arange(results["min_intensity"], results["max_intensity"])
    fit = results["model"].predict(x.reshape(-1, 1))
    ax.scatter(x, np.minimum(fit[-1]*2, results["variance"]), s=2, alpha=0.5)
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
    q = results['sensitivity']
    b = results['zero_level']
    v = ((movie[1:,:,:].astype('float64') - movie[:-1,:,:]) ** 2/2).mean(axis=0)
    imx = np.stack(((m-b)/q, v/q/q, (m-b)/q), axis=-1)
    im = ax.imshow(
        np.minimum(1, np.sqrt(0.01 + np.maximum(0, imx/np.quantile(imx, 0.9999))) - 0.1),
        cmap='PiYG'
    )
    _add_colorbar(ax, im, ticks=[0.2, .5, 0.8], labels=['<< 1', '1', '>> 1'])
    ax.axis(False)
    ax.set_title('Coefficient of Variation')
    ax.text(-0.1, 1.15, "C", transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='right')
    
    # D. Photon flux
    ax = next(axx)
    im = (movie.mean(axis=0)-results['zero_level'])/results['sensitivity']
    mx = np.quantile(im, 0.999)
    im = ax.imshow(im, vmin=-mx, vmax=mx, cmap=cc.cm.CET_D13)
    _add_colorbar(ax, im)
    ax.axis(False)
    ax.set_title('Photon Flux\n(photons/pixel/frame)')
    ax.text(-0.1, 1.15, "D", transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='right')
    
    if title:
        plt.suptitle(f'{title}\nPhoton sensitivity: {results["sensitivity"]:4.1f}')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
