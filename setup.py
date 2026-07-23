# Code attribution: Khrish Naik on YouTube, link: https://www.youtube.com/watch?v=S_F_c9e2bz4&list=PLZoTAELRMXVPS-dOaVbAux22vzqdgoGhG

from pathlib import Path
from setuptools import setup, find_packages
from typing import List


def get_requirements(file_path: str) -> List[str]:
    """
    Return a list of requirements from the given file.
    Removes '-e .' entries used for editable installs.
    """
    HYPHEN_E_DOT = "-e ."

    with open(file_path) as file_obj:
        requirements = file_obj.read().splitlines()
        requirements = [req.strip() for req in requirements if req and not req.startswith("#")]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="ml_fundamentals_practice",
    version="0.0.1",
    description="data processing and modelling utilities.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Diana Sulyma",
    author_email="sulymadi@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.13",
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    include_package_data=True,
)