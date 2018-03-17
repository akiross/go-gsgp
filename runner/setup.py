from setuptools import setup

setup(name='ugp_autorun',
      version='0.1',
      description='To run UGP and analyze its output',
      url='http://github.com/akiross/go-gsgp',
      author='Alessandro "AkiRoss" Re',
      author_email='ale@ale-re.net',
      license='GPL-3',
      packages=['autorun'],
      install_requires=[
          'statsmodels',
          'scipy',
          'numpy',
          'pandas',
          'matplotlib',
      ],
      zip_safe=False)
