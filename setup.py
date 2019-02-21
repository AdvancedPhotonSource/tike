#!/usr/bin/env python
# -*- coding: utf-8 -*-
from skbuild import setup
import setuptools

setup(
    name='tike',
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    setup_requires=['setuptools_scm'],
    use_scm_version=True,
    cmake_languages=('C'),
    include_package_data=True,
    zip_safe=False,
    author='Doga Gursoy',
    author_email='dgursoy@anl.gov',
    url='http://tike.readthedocs.org',
    download_url='http://github.com/tomography/tike.git',
    license='BSD-3',
    platforms='Any',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: C',
        'Topic :: Scientific/Engineering',
        ]
)
