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
   DAG.to_amat
   DAG.from_amat

Graph modification
------------------
.. autosummary::
   :toctree: generated

   DAG.add_node
   DAG.remove_node
   DAG.add_arc
   DAG.remove_arc
   DAG.add_nodes_from
   DAG.add_arcs_from

Graph properties
----------------
.. autosummary::
   :toctree: generated

   DAG.reversible_arcs
   DAG.vstructs

Comparison to other DAGs
------------------------
.. autosummary::
   :toctree: generated

   DAG.shd
   DAG.shd_skeleton
   DAG.markov_equivalent

Functions for nodes
-------------------
.. autosummary::
   :toctree: generated

   DAG.downstream
   DAG.upstream
   DAG.incident_arcs
   DAG.incoming_arcs
   DAG.outgoing_arcs
   DAG.indegree
   DAG.outdegree
