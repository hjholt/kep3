.. _propagation:

Numerical Propagation
#####################


The backbone of numerical propagation in `pykep` is based on Lagrangian coefficients for 
Kepler's  dynamics and Taylor numerical integration, as implemented in the 
`Heyoka <https://bluescarni.github.io/heyoka.py/index.html>`_ :cite:p:`biscaniheyoka1` python package, for all
other cases. The state transition matrix is also available and provided, in the case of numerical integration,
seamlessly via variational equations.

The main routines are listed here:
 
Keplerian 
******************
 
.. currentmodule:: pykep

.. autofunction:: propagate_lagrangian

.. autofunction:: propagate_lagrangian_grid

---------------------------------------------------------

Non-Keplerian 
************************

.. currentmodule:: pykep

.. autoclass:: stark_problem
    :members:  propagate, propagate_var, mu, veff, tol
