import setuptools
from pathlib import Path

import torch

setuptools.setup(
    name='gym_cricket',
    author="Andrea Rossolini",
    author_email="andrea.rossolini@ucalgary.ca",
    version='0.0.6',
    description="An OpenAI Gym Env for Cricket Robot @ UoC",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/...",
    packages=setuptools.find_packages(include="gym_panda*"),
    install_requires=['gym', 'pybullet', 'numpy', 'torch', 'pywavefront', 'matplotlib'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
    python_requires='>=3.6'
)