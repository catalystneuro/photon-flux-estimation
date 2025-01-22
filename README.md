# Photon Flux Estimation

A Python library for estimating photon flux from two-photon imaging data. This package provides tools to compute photon sensitivity and generate photon flux visualizations from imaging data.

## Installation

```bash
pip install photon-flux-estimation
```

## Features

- Compute photon sensitivity from imaging data
- Generate photon transfer curve visualizations
- Calculate photon flux estimates
- Visualize coefficient of variation
- Demo notebook with DANDI dataset integration

## Usage

```python
from photon_flux_estimation import PhotonFluxAnalyzer

# Load your imaging data (height, width, time)
movie = your_data_loading_function()

# Create analyzer and compute sensitivity
analyzer = PhotonFluxAnalyzer(movie)
results = analyzer.compute_sensitivity()

print(f"Sensitivity: {results['sensitivity']:.2f}")
print(f"Zero level: {results['zero_level']:.2f}")

# Get photon flux movie
photon_flux = analyzer.compute_photon_flux()

# Generate visualization
analyzer.plot_analysis(title="My Analysis")
```

## Citation

This package is based on code from the [compress-multiphoton](https://github.com/datajoint/compress-multiphoton) repository. If you use this package in your research, please cite both this package and the original repository.

## Documentation

For detailed documentation and examples, please see the [demo notebook](notebooks/demo.ipynb).

## License

MIT License. See [LICENSE](LICENSE) for details.
