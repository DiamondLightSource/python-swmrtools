from distutils.core import setup

setup(
    name="swmr_tools",
    packages=["swmr_tools"],
    version="0.4.0",
    license="MIT",
    description="Python iterator for safely monitoring NeXus files",
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
