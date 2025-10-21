import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="ICViT",
    py_modules=["icvit"],
    version="1.0",
    description="Isolated Channel Vision Transformers (IC-ViT)",
    author="anonymous",
    packages=find_packages() + ["icvit/config"],
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
)
