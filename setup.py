#!/usr/bin/env python
from setuptools import setup, find_packages

setup (
    name = "katfile",
    version = "trunk",
    description = "Karoo Array Telescope library to interact with HDF5 and MS files",
    author = "Ludwig Schwardt",
    author_email = "ludwig@ska.ac.za",
    packages = find_packages(),
    scripts = [
        "scripts/h5toms.py",
    ],
    url='http://ska.ac.za/',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    platforms = [ "OS Independent" ],
    keywords="kat kat7 ska",
    zip_safe = False,
    # Bitten Test Suite
    test_suite = "katfile.test.suite",
)
