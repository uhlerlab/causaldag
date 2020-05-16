.. _ancestral_graph:

**************
AncestralGraph
**************

Overview
********
.. currentmodule:: causaldag.classes.ancestral_graph
.. autoclass:: AncestralGraph

Methods
*******
.. autosummary::
   :toctree: generated

   AncestralGraph.copy
   AncestralGraph.to_amat
   AncestralGraph.from_amat

Graph modification
------------------
.. autosummary::
   :toctree: generated

   AncestralGraph.add_node
   AncestralGraph.remove_node
   AncestralGraph.add_directed
   AncestralGraph.remove_directed
   AncestralGraph.add_bidirected
   AncestralGraph.remove_bidirected
   AncestralGraph.add_undirected
   AncestralGraph.remove_undirected
   AncestralGraph.add_nodes_from
   AncestralGraph.remove_edge
   AncestralGraph.remove_edges

Graph properties
----------------
.. autosummary::
   :toctree: generated

   AncestralGraph.reversible_arcs
   AncestralGraph.vstructs

Comparison to other AncestralGraphs
------------------------
.. autosummary::
   :toctree: generated

   AncestralGraph.shd_skeleton
   AncestralGraph.markov_equivalent

Functions for nodes
-------------------
.. autosummary::
   :toctree: generated

   AncestralGraph.descendants_of
   AncestralGraph.ancestors_of
   AncestralGraph.parents_of
   AncestralGraph.children_of
   AncestralGraph.spouses_of
   AncestralGraph.neighbors_of
