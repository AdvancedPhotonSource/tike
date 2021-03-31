############
Contributing
############

Thank you for reading this. We're happy that you have decided to report a bug or
request a feature; your contribution will help make `tike` better.

**********
Philosophy
**********

Simple - Our contributors include non-computer scientists. Our code should be
easy to understand in order to lower the barrier to entry and promote
maintainability.

Suboptimal - Your time is precious. Save optimization for 10x improvements not
2x or 1.2x improvements.

Scalable - We need our code to scale for large data sets. Design should
consider multi-node applications.

***********
Report bugs
***********

Please open an Issue on our GitHub project page. Be sure to include the steps
that we can take to reproduce your bug. Be prepared for the bug hunting process
to take multiple days. We may ask for more information. Please be sure to close
the issue or let us know when the problem is resolved!

****************
Feature requests
****************

If you have an idea about how to improve our code, open an Issue on our GitHub
project page. Discussion about and planning for the best way to implement the
idea will reduce its development time.

*************
Pull Requests
*************

We want the `tike` codebase to be maintainable, simple, and lightweight. Please
expect pull requests to be reviewed with the following criterion in mind:

- Commit messages follow our guidelines.
- Documentation and tests are present.
- Variable names are explanatory.
- Code comments are used to clarify algorithms.
- Code structure is modular.
- Use of external dependencies is minimized.
- Code generally adheres to `PEP8 <https://www.python.org/dev/peps/pep-0008/#package-and-module-names>`_ style.

Commit messages
===============

Clear commit messages help us understand what and why changes were made. They
should follow the format below which we copied from the `NumPy development
workflow <https://docs.scipy.org/doc/numpy-1.15.0/dev/gitwash/development_workflow.html>`_ .

For example:

.. code-block:: none

  ENH: add functionality X to numpy.<submodule>.

  The first line of the commit message starts with a capitalized acronym
  (options listed below) indicating what type of commit this is. Then a blank
  line, then more text if needed.  Lines shouldn't be longer than 72
  characters.  If the commit is related to a ticket, indicate that with
  "See #3456", "See ticket 3456", "Closes #3456" or similar.

Standard acronyms to start the commit message with are:

.. code-block:: none

  API: an (incompatible) API change
  BENCH: changes to the benchmark suite
  BLD: change related to building numpy
  BUG: bug fix
  DEP: deprecate something, or remove a deprecated object
  DEV: development tool or utility
  DOC: documentation
  ENH: enhancement
  MAINT: maintenance commit (refactoring, typos, etc.)
  REV: revert an earlier commit
  STY: style fix (whitespace, PEP8)
  TST: addition or modification of tests
  REL: related to releasing numpy

Linting
=======

As part of our continuous integration tests, we `lint
<https://en.wikipedia.org/wiki/Lint_(software)>`_ our code using `pycodestyle
<https://github.com/PyCQA/pycodestyle>`_ and `pydocstyle
<https://github.com/PyCQA/pydocstyle>`_.
