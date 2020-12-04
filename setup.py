from os import path

from setuptools import setup

setup(name='marlenv',
      version='0.0.1',
      url='https://github.com/kc-ml2/marlenv',
      py_modules=['marlenv'],
      author='Tae Min Ha, Daniel Nam',
      author_email='taemin410@gmail.com, dwtnam@kc-ml2.com',
      # license=open(path.join(path.abspath(path.dirname(__file__)), 'LICENSE')).read(),
      install_requires=[x.strip() for x in
                        open(path.join(path.abspath(path.dirname(__file__)), 'requirements.txt')).readlines()],
      python_requires='>=3.7',
      )
