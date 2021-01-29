from causaldag.utils.ci_tests import partial_correlation_suffstat

def compute_gaussian_interventional_suffstat(intervention_info):
    return {intervened_nodes: partial_correlation_suffstat(samples) for intervened_nodes, samples in intervention_info.items()}