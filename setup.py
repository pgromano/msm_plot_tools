from setuptools import setup, Extension, find_packages
from os.path import join
import numpy

setup(
    name = 'plot_tools',
    author = 'Pablo Romano',
    author_email = 'promano@uoregon.edu',
    description = 'Python API for Plotting MSM',
    version = '0.1',

    packages = ['plot_tools'],
    install_requires=[
        'matplotlib',
        'numpy',
        'scikit-learn'
    ],
    zip_safe = False,
)
