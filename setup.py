try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='tools',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description="Environment for Team creek.",
      long_description="Environment for team creek.",
      url='https://github.com/ese-msc-2021/ads-wildfire-team-creek',
      author="Mew, Zoe, Theo, Olga, Shuge, Zhijie",
      packages=['tools'])
