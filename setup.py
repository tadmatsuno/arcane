"""Setup script for the arcane package."""

from pathlib import Path
from typing import List
import os

from setuptools import find_packages, setup


ROOT = Path(__file__).parent.resolve()


def _read_requirements(filename: str) -> List[str]:
    """Load dependency list from requirements.txt while ignoring comments."""

    req_path = ROOT / filename
    if not req_path.exists():
        return []

    requirements: List[str] = []
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


if not os.path.exists(os.path.expanduser("~/.arcanesrc")):
    with open(os.path.expanduser("~/.arcanesrc"), "w", encoding="utf-8") as f:
        f.write("{\n\n}")


setup(
    name="arcane",
    version="0.0.1",
    description="Tools for high-resolution stellar abundance analysis",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="T. Matsuno",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=_read_requirements("requirements.txt"),
    include_package_data=True,
    package_data={
        "arcane": [
            "continuum/*.ui",
            "ew/*.ui",
        ]
    },
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
)
