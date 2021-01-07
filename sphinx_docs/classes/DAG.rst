.. _dag:

***
DAG
***

.. currentmodule:: causaldag.classes.dag

Copying
-------
.. autosummary::
   :toctree: generated

   DAG.copy
   DAG.rename_nodes
   DAG.induced_subgraph

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

Ordering
--------
.. autosummary::
   :toctree: generated

   DAG.topological_sort
   DAG.is_topological
   DAG.permutation_score

Comparison to other DAGs
------------------------
.. autosummary::
   :toctree: generated

   DAG.shd
   DAG.shd_skeleton
   DAG.markov_equivalent
   DAG.is_imap
   DAG.is_minimal_imap
   DAG.chickering_distance
   DAG.confusion_matrix
   DAG.confusion_matrix_skeleton

Separation statements
---------------------
.. autosummary::
   :toctree: generated

   DAG.dsep
   DAG.dsep_from_given
   DAG.is_invariant
   DAG.local_markov_statements

Conversion to other formats
---------------------------
.. autosummary::
   :toctree: generated

   DAG.from_amat
   DAG.to_amat
   DAG.from_nx
   DAG.to_nx
   DAG.from_dataframe
   DAG.to_dataframe

Conversion to other graphs
--------------------------
.. autosummary::
   :toctree: generated

   DAG.moral_graph
   DAG.marginal_mag
   DAG.cpdag
   DAG.interventional_cpdag

Chickering Sequences
--------------------
.. autosummary::
   :toctree: generated

   DAG.resolved_sinks
   DAG.chickering_sequence
   DAG.apply_edge_operation

Directed Clique Trees
---------------------
.. autosummary::
   :toctree: generated

   DAG.directed_clique_tree
   DAG.simplified_directed_clique_tree
   DAG.residuals
   DAG.residual_essential_graph

Intervention Design
-------------------
.. autosummary::
   :toctree: generated

   DAG.optimal_fully_orienting_single_node_interventions
   DAG.greedy_optimal_single_node_intervention
   DAG.greedy_optimal_fully_orienting_interventions
