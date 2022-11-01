from distutils.core import setup

# read the contents of your README file
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="swmr_tools",
    packages=["swmr_tools"],
    version="0.7.3",
    license="MIT",
    description="Python iterator for safely monitoring NeXus files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Diamond Light Source Ltd",
    author_email="scientificsoftware@diamond.ac.uk",
    url="https://github.com/DiamondLightSource/python-swmrtools",
    keywords=["HDF5", "Iterator", "Diamond"],
    install_requires=["numpy", "h5py"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
)
