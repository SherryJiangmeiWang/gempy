.. GeMpy documentation master file, created by
   sphinx-quickstart on Wed Dec 14 12:44:40 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GemPy's (v 1.0) documentation
=============================
Software for 3D structural geologic implicit modeling in Python.
****************************************************************

GemPy is an open-source tool for generating 3D structural geological models in Python (GemPy's code can be viewed in its repository: https://github.com/cgre-aachen/gempy.)
It is capable of creating complex 3D geological models,
including stratigraphic and structural features such as:

- fold structures (e.g.: anticlines, synclines)
- fault networks and fault-layer interactions
- unconformities

3D models created with GemPy may look like this:

.. image:: ./images/model_example.png

Contents:

.. toctree::
   :maxdepth: 2

   self
   ./theory/motivation
   tutorial
   theory
   code

The core algorithm is based on a universal cokriging interpolation method devised by
Lajaunie et al. (1997) and further elaborated by Calcagno et al. (2008).
Its implicit nature allows the user to generate complete 3D geological models
through the interpolation of input data consisting of:

- Surface contact points: 3D coordinates of points marking the boundaries between different features.
- Orientation measurements: Orientation of the poles perpendicular to the dipping of surfaces at any point in the 3D space.

GemPy also allows for the definition of topological elements such as stratigraphic sequences and fault networks to be considered in this process.

.. image:: ./images/modeling_principle.png

GemPy itself offers direct visualization of 2D sections via matplotlib
and in full 3D using the Visualization Toolkit (VTK). These VTK files can also be exported
for further processing in programs such as Paraview. GemPy can furthermore be easily
embedded in Blender for 3D rendering.
Another option is Steno3D, which allows for a flexible and interactive visualization of 3D models:

.. raw:: html

    <div style="margin-top:10px;">
      <iframe src="https://steno3d.com/embed/mWACfOTPB2vT3scgJABp" width="800" height="600"></iframe>
    </div>



GemPy was furthermore designed to allow the performance of
Bayesian inference for stochastic geological modeling. This was achieved by writing GemPy's core algorithm
in Theano (http://deeplearning.net/software/theano/) and coupling it with PyMC3 (https://pymc-devs.github.io/pymc3/notebooks/getting_started.html).
This enables the use of advanced HMC methods and is of particular relevance when considering
uncertainties in model input data and the availability of additional secondary information.

We can, for example, include uncertainties with respect to the z-position of layer boundaries
in the model space. Simple Monte Carlo simulation via PyMC will then result in different model realizations:

|z_unc| |wobble|

.. |z_unc| image:: ./images/gempy_zunc.png
   :width: 40%

.. |wobble| image:: ./images/model_wobble.gif
   :width: 55%

This opens the path to...

((This optimization package allows the computation
of gradients opening the door to the use of advance HMC methods
coupling GeMpy and PyMC3 (https://pymc-devs.github.io/pymc3/notebooks/getting_started.html).
Also, the use of theano allows the use of the GPU through cuda (see theano doc for more information).)

For a more detailed elaboration of the theory behind GemPy, take a look at the upcoming scientific publication
"GemPy 1.0: open-source stochastic geological modeling and inversion" by de la Varga et al. (2018).


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
