.. _pdag:

****
PDAG
****

Overview
********
.. currentmodule:: causaldag.classes.pdag
.. autoclass:: PDAG

Methods
*******
.. autosummary::
   :toctree: generated

   PDAG.copy
   PDAG.to_amat
   PDAG.from_amat

Graph modification
------------------
.. autosummary::
   :toctree: generated

   PDAG.remove_node
   PDAG.add_known_arc

Graph properties
----------------
.. autosummary::
   :toctree: generated

   PDAG.has_edge
   PDAG.has_edge_or_arc

Comparison to other PDAGs
-------------------------
.. autosummary::
   :toctree: generated

   PDAG.shd

Functions for
-------------------
.. autosummary::
   :toctree: generated

   PDAG.to_dag
   PDAG.all_dags
