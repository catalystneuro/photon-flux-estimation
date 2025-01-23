from setuptools import setup, find_packages

setup(
    name="photon-flux-estimation",
    version="0.1.0",
    author="Alessandra Trapani",
    author_email='alessandra.trapani@catalystneuro.com',
    description="Library to compute estimated photon flux from multiphoton imaging data",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/catalystneuro/photon-flux-estimation',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
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
