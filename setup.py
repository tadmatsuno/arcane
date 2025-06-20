from setuptools import setup, find_packages
import os

if not os.path.exists(os.path.expanduser("~/.arcanesrc")):
    with open(os.path.expanduser("~/.arcanesrc"),"w") as f:
        f.write("{\n\n}")
    

setup(
    name='arcane',
    version='0.0.1',
    description='a prototype of my arcane package',
    packages=find_packages(),  # Automatically discover packages and subpackages
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'PyQt5',
        'pandas',
        'shapely',
        'astropy',
        'tqdm',
    ],
)
