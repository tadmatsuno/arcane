from setuptools import setup, find_packages

setup(
    name='arcane',
    version='0.0.1',
    description='a plot type of my arcane package',
    packages=find_packages(),  # Automatically discover packages and subpackages
    install_requires=[
        # List your package dependencies here, e.g., 'requests>=2.0'
    ],
)