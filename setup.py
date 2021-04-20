from os import path
from setuptools import find_packages, setup


CURRENT_DIR = path.abspath(path.dirname(__file__))


def read_me(filename):
    with open(path.join(CURRENT_DIR, filename), encoding='utf-8') as f:
        return f.read()


def requirements(filename):
    with open(path.join(CURRENT_DIR, filename)) as f:
        return f.read().splitlines()


AUTHORS = ""
NAME = "laplace"
PACKAGES = find_packages()
DESCR = ""
LONG_DESCR = ""
LONG_DESCR_TYPE = 'text/markdown'
REQUIREMENTS = requirements('requirements.txt')
VERSION = "0.1"
URL = ""
LICENSE = ""


setup(
    author=AUTHORS,
    name=NAME,
    version=VERSION,
    description=DESCR,
    long_description=LONG_DESCR,
    long_description_content_type=LONG_DESCR_TYPE,
    install_requires=REQUIREMENTS,
    url=URL,
    license=LICENSE,
    packages=PACKAGES,
    zip_safe=False,
    python_requires=">=3.7",
)
