from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

with open(path.join(here, "requirements_common.txt"), encoding="utf-8") as f:
    common_requirements = [req.strip() for req in f if req]

with open(path.join(here, "requirements_serve.txt"), encoding="utf-8") as f:
    serve_requirements = [req.strip() for req in f if req]

setup(
    name="ml_workshope",
    version="0.0.1",
    description="A small workshop package",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Zhong Qishuai",
    author_email="ferdinandzhong@gmail.com",
    packages=find_packages(include=["workshop"]),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">3.9",
    install_requires=common_requirements,
    extras_require={
        "serve": serve_requirements,
    },
    zip_safe=False,
    entry_points={
        "console_scripts": [],
    },
)