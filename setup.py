from setuptools import setup, find_packages

setup(
    name="photon-flux-estimation",
    version="0.1.0",
    description="Library to compute estimated photon flux from two-photon imaging data",
    author="CatalystNeuro",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "matplotlib",
        "pynwb",
        "dandi",
        "h5py",
        "colorcet",
        "fsspec",
    ],
    python_requires=">=3.8",
)
