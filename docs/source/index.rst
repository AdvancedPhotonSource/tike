.. automodule:: tike

  ************
  Key Concepts
  ************
  The key concepts of the Tike code structure are the following:

   1. Solvers are based on forward and adjoint operators

   2. Solvers are organized into modules based on problem-type


  Solvers share fundamental operators
  ===================================
  An operator is a transforms data from one space to another. For example, the
  tomography forward operator is the radon transform; it maps a 2D Cartesian space
  to the sinogram space. The adjoint operator maps data from sinogram space back
  to 2d Cartesian. All iterative solvers of a problem use forward and adjoint, so
  we avoid code duplication by implementing all solvers in terms of forward and
  adjoint operators to reduce code duplication to make future-upgrades easier.


  Modules by problem type
  =======================
  The solvers for each problem-type (ptychography, tomography, etc) are separated
  into modules of their respective names.


.. toctree::
   :maxdepth: 3
   :hidden:

   api
