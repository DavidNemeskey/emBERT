#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

# I used the following resources to compile the packaging boilerplate:
# https://python-packaging.readthedocs.io/en/latest/
# https://packaging.python.org/distributing/#requirements-for-packaging-and-distributing
import sys

from setuptools import find_packages, setup

def readme():
    with open('README.md') as f:
        return f.read()

# A few things depend on the Python version
version = '.'.join(map(str, sys.version_info))
if version < '3.6':
    raise ValueError('The oldest Python version supported is 3.6.')
dataclasses_req = ['dataclasses==0.7'] if version < '3.7' else []


setup(name='embert',
      version='1.0.5',
      description='A Python package for integrating BERT-based NLP models '
                  'into emtsv. Also provides scripts for training and '
                  'analyzing them.',
      long_description=readme(),
      url='https://github.com/DavidNemeskey/emBERT',
      author='Dávid Márk Nemeskey',
      license='LGPL',
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 5 - Stable',

          # Indicate who your project is intended for
          'Intended Audience :: Science/Research',
          # This one is not in the list...
          'Topic :: Scientific/Engineering :: Natural Language Processing',

          # Environment
          'Operating System :: POSIX :: Linux',
          'Environment :: Console',
          'Natural Language :: English',

          # Pick your license as you wish (should match "license" above)
          'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7'
      ],
      keywords='BERT transformer NER chunking',
      packages=find_packages(exclude=['scripts']),
      # Install the scripts
      scripts=[
          'scripts/split_to_sets.py',
          'scripts/tokenization_comparison.py',
          'scripts/train_embert.py',
      ],
      install_requires=dataclasses_req + [
          'progressbar',
          'pygithub',
          'pyyaml',
          'requests',
          'seqeval<=0.0.5',
          'torch<1.5.0',  # torch 1.5 is buggy
          'tqdm',
          'transformers',
      ],
      # zip_safe=False,
      use_2to3=False)
