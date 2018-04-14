from setuptools import find_packages
from setuptools import setup

setup(name='USI-RNN',
      version='0.0.1',
      description='Modeling Unevenly Spaced Intervals',
      author='Dmitry Martyanov',
      author_email='dmitry.a.martyanov@gmail.com',
      url='https://github.com/dmartyanov/USI-RNN',
      download_url='https://github.com/dmartyanov/USI-RNN/tarball/0.0.1',
      license='MIT',
      install_requires=['Keras==2.0.4'],
      packages=find_packages())