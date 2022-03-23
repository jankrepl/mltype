from setuptools import find_packages, setup

import mltype

DESCRIPTION = "Command line tool for improving typing skills"
LONG_DESCRIPTION = DESCRIPTION

INSTALL_REQUIRES = [
    "click",
    "click-config-file",
    "numpy",
    "torch",
    "pytorch-lightning>=1.0.0,<1.2",
    "tqdm",
]

setup(
    name="mltype",
    version=mltype.__version__,
    author="Jan Krepl",
    author_email="kjan.official@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url="https://github.com/jankrepl/mltype",
    packages=find_packages(exclude=["tests"]),
    license="MIT",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": [
            "black",
            "flake8",
            "pydocstyle",
            "pytest",
            "pytest-coverage",
            "tox",
        ],
        "mlflow": ["mlflow<=1.10.0"],
    },
    entry_points={
        "console_scripts": [
            "mlt = mltype.cli:cli",
        ]
    },
)
