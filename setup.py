from setuptools import setup, find_packages

setup(
    name="temporal_composition_gym",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "minigrid",
        "pygame",
        "pymunk",
        "numpy",
        "pytablericons",
        "Pillow",
    ],
)
