####
Tike
####

.. image:: https://dev.azure.com/tomopy/tike/_apis/build/status/tomography.tike?branchName=master
   :target: https://dev.azure.com/tomopy/tike/_build/latest?definitionId=3&branchName=master
   :alt: Build Status

Tike is a toolbox for tomographic reconstruction of 3D objects from ptychography
data.

The aim of Tike is to provide fast and accurate implementations of a wide
variety of reconstruction algorithms, and to provide a common platform for the
synchrotron research community.

**************************
Current Features (213c543)
**************************

Scan
====
- Lissajous and 2D spiral trajectories
- hexagonal and rectangular grids

Ptychography
============

- FFT-based operator with linear position interpolation
- single-energy
- multiple scan positions per diffraction pattern (fly-scans)
- multiple incoherent modes per diffraction pattern (multi-mode probes)
- one shared (multi-modal) probe per angular view
- conjugate-gradient descent solver
- least-squares + gradient descent solver

Laminography
============

- USFFT-based operator for cubic field-of-view
- single tilt angle
- conjugate-gradient descent solver

Alighnment
==========
- Lanczos-based rotation and flow operators
