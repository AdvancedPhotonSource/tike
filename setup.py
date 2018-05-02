#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, Extension, find_packages
import os

tomoc = Extension(
    name='tike.libtike',
    sources=['src/tomo.c'])

ext_mods = [tomoc]

# Remove external C code for RTD builds
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    ext_mods = []

setup(
    name='tike',
    packages=find_packages(exclude=['tests*']),
    version=open('VERSION').read().strip(),
    include_package_data=True,
    ext_modules=ext_mods,
    zip_safe=False,
    author='Doga Gursoy',
    author_email='dgursoy@anl.gov',
    url='http://tike.readthedocs.org',
    download_url='http://github.com/dgursoy/tike.git',
    license='BSD-3',
    platforms='Any',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: C']
)
