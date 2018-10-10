#!/usr/bin/env python

################################################################################
# Copyright (c) 2011-2018, National Research Foundation (Square Kilometre Array)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

from __future__ import print_function, division, absolute_import

import os.path

from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))
readme = open(os.path.join(here, 'README.rst')).read()
news = open(os.path.join(here, 'NEWS.rst')).read()
long_description = readme + '\n\n' + news

setup(name='katdal',
      description='Karoo Array Telescope data access library for interacting '
                  'with data sets in the MeerKAT Visibility Format (MVF)',
      long_description=long_description,
      author='Ludwig Schwardt',
      author_email='ludwig@ska.ac.za',
      packages=find_packages(),
      scripts=[
          'scripts/h5list.py',
          'scripts/h5toms.py',
          'scripts/mvftoms.py',
          'scripts/fix_ant_positions.py'],
      url='https://github.com/ska-sa/katdal',
      license='Modified BSD',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Topic :: Software Development :: Libraries :: Python Modules',
          'Topic :: Scientific/Engineering :: Astronomy'],
      platforms=['OS Independent'],
      keywords='meerkat ska',
      python_requires='>=2.7,!=3.0,!=3.1,!=3.2',
      setup_requires=['katversion'],
      use_katversion=True,
      install_requires=['numpy', 'katpoint', 'h5py>=2.3',
                        'katsdptelstate[rdb]', 'dask[array] >= 0.18.2', 'numba',
                        'requests >= 2.18.0', 'defusedxml', 'future'],
      extras_require={
          'ms': ['python-casacore >= 2.2.1'],
          's3': [],
          's3credentials': ['botocore'],
          # rados is not in PyPI but available as Debian package python-rados
          'rados': ['rados']
      },
      tests_require=['mock', 'nose', 'subprocess32'])
