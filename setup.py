from os import path

from setuptools import setup

setup(name='ml2_marlenv',
      version='0.0.1',
      url='https://github.com/taemin410/ml2_marlenv',
      py_modules=['ml2_marlenv'],
      author='Tae Min Ha',
      author_email='taemin410@gmail.com',
      # license=open(path.join(path.abspath(path.dirname(__file__)), 'LICENSE')).read(),
      install_requires=[x.strip() for x in
                        open(path.join(path.abspath(path.dirname(__file__)), 'requirements.txt')).readlines()],
      python_requires='>=3.5',
      )
