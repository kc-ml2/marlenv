from setuptools import setup, find_packages

setup(
    name='marlenv',
    packages=[package for package in find_packages() if package.startswith('marlenv')],
    version='1.0.0',
    url='https://github.com/kc-ml2/marlenv',
    author='Tae Min Ha, Daniel Nam',
    author_email='contact@kc-ml2.com',
    install_requires=[],
    python_requires='>=3.7'
)