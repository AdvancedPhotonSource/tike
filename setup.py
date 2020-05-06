#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools

setuptools.setup(
    name='tike',
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
    use_scm_version=True,
    author='Doga Gursoy',
    author_email='dgursoy@anl.gov',
    url='http://tike.readthedocs.org',
    download_url='http://github.com/tomography/tike.git',
    license='BSD-3',
    platforms='Any',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        ],
    entry_points={
        'tike.Ptycho': [
            'cupy = tike.operators.cupy:Ptycho',
            'numpy = tike.operators.numpy:Ptycho',
        ],
        'tike.Convolution': [
            'cupy = tike.operators.cupy:Convolution',
            'numpy = tike.operators.numpy:Convolution',
        ],
        'tike.Propagation': [
            'cupy = tike.operators.cupy:Propagation',
            'numpy = tike.operators.numpy:Propagation',
        ],
    },
)
