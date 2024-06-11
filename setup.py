from setuptools import setup

from Cython.Build import cythonize
import numpy as np

PACKAGE_DIR = "pypls"

setup(
    name="pypls",
    version="1.0.3",
    description="pypls: A package for statistical analysis using PLS.",
    author="Dong Nai-ping",
    author_email="naiping.dong@hotmail.com",
    packages=[
        "core"
    ],
    ext_modules=cythonize(
        [
            "core/*.pyx"
        ],
        compiler_directives={
            "language_level": "3",
        }
    ),
    include_dirs=[
        np.get_include()
    ],
    include_package_data=True,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "License :: OSI Approved :: Apache License v2",
        "Natural Language :: English"
    ],
)
