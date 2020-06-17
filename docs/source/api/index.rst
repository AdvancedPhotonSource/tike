#############
API Reference
#############

This part of the documentation describes each function, class, and method in
detail.

***************
Class structure
***************

The chart below is a `class diagram
<https://en.wikipedia.org/wiki/Class_diagram>`_. It shows the inheritance and
composition relationships between the classes. For example, :py:class:`Ptycho`
is an :py:class:`Operator` which is composed of :py:class:`Convolution` and
:py:class:`Propagation` (which are also Operators).

.. The class diagram is manually rendered using mermaidjs and saved as an svg
   the sphinxcontrib-mermaid package is out of date and doesn't work (21.3.2020)

.. image:: api-class-diagram.svg

*******
Modules
*******

The modules in Tike are as follows:

.. toctree::
   :maxdepth: 1

   operators
   opt
   ptycho
   lamino
   view
