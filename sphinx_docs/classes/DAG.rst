.. _dag:

***
DAG
***

Overview
********
.. currentmodule:: causaldag.classes.dag
.. autoclass:: DAG

Methods
*******
.. autosummary::
   :toctree: generated

   DAG.copy

Information about nodes
-----------------------
.. autosummary::
   :toctree: generated

   DAG.parents_of
   DAG.children_of
   DAG.neighbors_of
   DAG.markov_blanket_of
   DAG.ancestors_of
   DAG.descendants_of
   DAG.indegree_of
   DAG.outdegree_of
   DAG.incoming_arcs
   DAG.outgoing_arcs
   DAG.incident_arcs

Graph modification
------------------
.. autosummary::
   :toctree: generated

   DAG.add_node
   DAG.add_nodes_from
   DAG.remove_node
   DAG.add_arc
   DAG.add_arcs_from
   DAG.remove_arc
   DAG.reverse_arc

Graph properties
----------------
.. autosummary::
   :toctree: generated

   DAG.has_arc
   DAG.sources
   DAG.sinks
   DAG.reversible_arcs
   DAG.is_reversible
   DAG.arcs_in_vstructures
   DAG.vstructures
   DAG.triples
   DAG.upstream_most

Comparison to other DAGs
------------------------
.. autosummary::
   :toctree: generated

   DAG.shd
   DAG.shd_skeleton
   DAG.markov_equivalent

Separation statements
---------------------
.. autosummary::
   :toctree: generated

   DAG.dsep
   DAG.dsep_from_given

Conversion
**********
.. autosummary::
   :toctree: generated

   DAG.from_amat
   DAG.to_amat
   DAG.from_nx
   DAG.to_nx
   DAG.from_dataframe
   DAG.to_dataframe