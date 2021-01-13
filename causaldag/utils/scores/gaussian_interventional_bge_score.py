import numpy as np
# import numba
import scipy as sp
from scipy import stats
from scipy.special import loggamma
import math
# import ipdb

# @numba.jit
def numba_inv(A):
    return np.linalg.inv(A)

def faster_inverse(A):
    n = A.shape[0]
    b = np.eye(n)
    _, _, x, _ = sp.linalg.lapack.dgesv(A, b, 0, 0)
    
    return x

def chol_sample(mean, cov):
    return mean + np.linalg.cholesky(cov) @ np.random.standard_normal(mean.size)

def get_complete_dag(n):
    dag_incidence = np.ones((n, n))
    return np.triu(dag_incidence, 1)

def process_interventions(variables, interventions):
    """
    Processses interventions to a  

    Parameters
    ----------
    variables:
        TODO - describe.
    interventions:
        TODO - describe.

    Returns
    -------
    dict objects with keys: 'intervened_nodes', 'intervened_incidences', 'intervened_weights', 'intervened_biases', 'intervened_variances'
    """
    num_vars = len(variables)
    num_interventions = len(interventions)
    var_map = {variables[i] : i for i in range(num_vars)}
    # print(variables)
    # intervened_nodes intervened_nodes containing 1 iff that node is intervened
    intervened_nodes = np.zeros((num_interventions, num_vars))

    # intervened_incidences will contain 1 iff the edge weight is intervened on
    intervened_incidences = np.zeros((num_interventions, num_vars, num_vars))

    # intervened_weights will contain edge weights for all interventions
    intervened_weights = np.zeros((num_interventions, num_vars, num_vars))

    # intervened_biases will contain all biases from interventions
    intervened_biases = np.zeros((num_interventions, num_vars))

    # intervened_variances will contain all variances from interventions
    intervened_variances = np.zeros((num_interventions, num_vars))

    for j in range(num_interventions):
        intervention = interventions[j]
        intervened_incidence = np.zeros((num_vars, num_vars))
        for i in range(len(intervention)):
            each_intervention = intervention[i]
            if len(each_intervention) > 0 and each_intervention['node'] in var_map:
                node_idx = var_map[each_intervention['node']]
                intervened_nodes[j, node_idx] = 1
                intervened_biases[j, node_idx] = each_intervention['new_bias']
                intervened_variances[j, node_idx] = each_intervention['new_variance']

                for parent_and_weights in each_intervention['new_parent_coefficients']:
                    intervened_weights[j, parent_and_weights[0], node_idx] = parent_and_weights[1]
                    intervened_incidence[parent_and_weights[0], node_idx] = 1
        
        intervened_incidences[j] = intervened_incidence
    
    output = {'intervened_nodes' : intervened_nodes,
            'intervened_incidences' : intervened_incidences, 
            'intervened_weights': intervened_weights, 
            'intervened_biases' : intervened_biases,
            'intervened_variances' : intervened_variances}
    # print(output)
    return output


def var_set_monte_carlo_interventional_bge_score(
        variables,
        samples,
        processed_interventions,
        incidence,
        alpha_mu=None,
        alpha_w=None,
        inverse_scale_matrix=None,
        parameter_mean=None,
        is_diagonal=True,
        num_iterations=100
):
    if not is_diagonal:
        raise NotImplementedError("BGE score not implemented for non-diagonal matrix.")
    
    _, p = np.shape(samples)
    num_vars_monte_carlo = len(variables)
    V = list(variables)
    I = np.eye(num_vars_monte_carlo)

    if alpha_mu is None:
        alpha_mu = p
    if alpha_w is None:
        alpha_w = p + alpha_mu + 1
    if inverse_scale_matrix is None:
        inverse_scale_matrix = np.eye(p) * alpha_mu * (alpha_w - p - 1) / (alpha_mu + 1)
    if parameter_mean is None:
        parameter_mean = np.zeros(p)

    scale_matrix = faster_inverse(inverse_scale_matrix)
    df = alpha_w
    parameter_mean_monte_carlo = parameter_mean[V]
    scale_matrix_monte_carlo = scale_matrix[V, :][:, V]
    if num_iterations == 0 or len(variables) == 0:
        return 0
    
    standard_normal = lambda t: np.random.normal(0, 1)
    vfunc_standard_normal = np.vectorize(standard_normal)
    c_squared = np.zeros((num_iterations, num_vars_monte_carlo))
    
    for var in range(num_vars_monte_carlo):
        c_squared[:, var] = stats.chi2.rvs(df - p + var + 1, size = num_iterations)
        
    c = np.sqrt(c_squared)
    indices = np.where(incidence == 1)
    inverse_c = 1/np.array(c)
    num_samples = len(samples)
    # print(processed_interventions['intervened_incidences'].shape)
    assert(num_samples == np.shape(processed_interventions['intervened_incidences'])[0])

    def monte_carlo_iteration(iteration):
        nonlocal parameter_mean_monte_carlo
        B = np.zeros((num_vars_monte_carlo, num_vars_monte_carlo))
        
        if len(B[indices]) > 0:
            B[indices] = vfunc_standard_normal(B[indices])
        
        B = np.multiply(-np.array(B), inverse_c[iteration])
        d = np.zeros((num_vars_monte_carlo, num_vars_monte_carlo))
        np.fill_diagonal(d, scale_matrix_monte_carlo @ c_squared[iteration])
        # Compute from formula
        A = I - B.T
        inverse_sigma = A.T @ d @ A
        sigma = faster_inverse(inverse_sigma)
        mu_covariance = (1/alpha_mu) * sigma
        mu = chol_sample(parameter_mean_monte_carlo, mu_covariance) 
        log_likelihood_sum = 0
        monte_carlo_samples = samples[:, V]
        inverse_d = faster_inverse(d)

        for i in range(num_samples):
            # print(processed_interventions['intervened_nodes'][i])
            new_B = np.copy(B)
            variances = inverse_d
            data_point = monte_carlo_samples[i]

            intervened_indices = np.where(processed_interventions['intervened_incidences'][i] > 0)
            new_B[intervened_indices] = processed_interventions['intervened_weights'][i][intervened_indices]
            intervened_nodes = np.where(processed_interventions['intervened_nodes'][i] > 0)[0]
            variances[np.ix_(intervened_nodes, intervened_nodes)] = processed_interventions['intervened_variances'][i][intervened_nodes]
            mu[intervened_nodes] = processed_interventions['intervened_biases'][i][intervened_nodes]
            x_epsilon = (data_point - mu) - np.dot(new_B, data_point - mu)
            
            # Possible that there were hard interventions
            nonzero_variance_indices = np.where(np.diagonal(variances) > 0)[0]
            parameter_mean_monte_carlo = parameter_mean_monte_carlo[nonzero_variance_indices]
            variances = variances[np.ix_(nonzero_variance_indices, nonzero_variance_indices)]
            x_epsilon = x_epsilon[nonzero_variance_indices]
            prob_x = stats.multivariate_normal.logpdf(x_epsilon, parameter_mean_monte_carlo, variances)
            log_likelihood_sum += prob_x
        
        return log_likelihood_sum

    log_marginal_likes = list(map(monte_carlo_iteration, range(num_iterations)))
    log_marginal_likes_logsumexp = (sp.special.logsumexp(log_marginal_likes) - np.log(num_iterations)) 

    return log_marginal_likes_logsumexp

def local_gaussian_monte_carlo_bge_score(
        node,
        parents,
        samples,
        interventions=None,
        alpha_mu=None,
        alpha_w=None,
        inverse_scale_matrix=None,
        parameter_mean=None,
        is_diagonal=True,
        num_iterations=1000
):
    """
    Compute the BGE score of a node given its parents.

    Parameters
    ----------
    node:
        TODO - describe.
    parents:
        Topologically sorted list of parents for a node.
    samples:
        TODO - describe.
    interventions:
        List if list of dicts with keys 'node', 'new_parent_coefficients' 'new_bias', 'new_variance'
        * interventions[i][j]['node'] = node of intervention
        * interventions[i][j]['new_parent_coefficients'] = list of (pa_j, c_j) where pa_j is a parent of j
        * interventions[i][j]['new_bias'] = bias 
        * interventions[i][j]['new_variance'] = new_variance
    alpha_mu:
        TODO - describe. Default is the number of variables.
    alpha_w:
        TODO - describe. Default is the (number of variables) + alpha_mu + 1
    inverse_scale_matrix:
        TODO - describe. Default is the identity matrix.
    parameter_mean:
        TODO - describe. Default is the zero vector.
    is_diagonal:
        TODO - describe.

    Returns
    -------
    float
        BGE score.
    """
    if not is_diagonal:
        raise NotImplementedError("BGE score not implemented for non-diagonal matrix.")

    k = len(parents)
    _, p = np.shape(samples)
    
    if interventions is None:
        interventions = [[] for i in range(len(samples))]
    if alpha_mu is None:
        alpha_mu = p
    if alpha_w is None:
        alpha_w = p + alpha_mu + 1
    if inverse_scale_matrix is None:
        inverse_scale_matrix = np.eye(p) * alpha_mu * (alpha_w - p - 1) / (alpha_mu + 1)
    if parameter_mean is None:
        parameter_mean = np.zeros(p)

    list_parents_and_node = [*parents, node]
    default_incidence = get_complete_dag(k+1)
    processed_interventions = process_interventions(list_parents_and_node, interventions)

    ### First, compute for numerator p(d^{Pa_i U {X_i}} | m^h) ###
    variables = [i for i in range(k + 1)]
    marginal_likelihood_parents_and_node = var_set_monte_carlo_interventional_bge_score(variables, samples, processed_interventions, default_incidence, alpha_mu, alpha_w, inverse_scale_matrix, parameter_mean, is_diagonal, num_iterations)
    print("marginal_likelihood_parents_and_node", marginal_likelihood_parents_and_node)
    
    # 'intervened_nodes', 'intervened_incidences', 'intervened_weights', 'intervened_biases', 'intervened_variances'
    print("-----")
    new_processed_interventions = {}
    new_processed_interventions['intervened_nodes'] = processed_interventions['intervened_nodes'][:, :-1]
    new_processed_interventions['intervened_incidences'] = processed_interventions['intervened_incidences'][:, :-1, :-1]
    new_processed_interventions['intervened_weights'] = processed_interventions['intervened_weights'][:, :-1, :-1]
    new_processed_interventions['intervened_biases'] = processed_interventions['intervened_biases'][:, :-1]
    new_processed_interventions['intervened_variances'] = processed_interventions['intervened_variances'][:, :-1]

    ### First, compute for numerator p(d^{Pa_i | m^h) ###
    variables = variables[:k]
    marginal_likelihood_parents = var_set_monte_carlo_interventional_bge_score(variables, samples, new_processed_interventions, default_incidence[:k, :k], alpha_mu, alpha_w, inverse_scale_matrix, parameter_mean, is_diagonal, num_iterations)
    print("marginal_likelihood_parents", marginal_likelihood_parents)

    return (marginal_likelihood_parents_and_node - marginal_likelihood_parents)

if __name__ == '__main__':
    gaussian_data = np.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]])
    # interventions = [[{'node' : 1, 'new_parent_coefficients' : [(0, 0.5)], 'new_bias' : 0.3, 'new_variance': 1.0}], []]
    interventions = [[], []]
    s1 = local_gaussian_monte_carlo_bge_score(0, set(), gaussian_data, interventions)
    s2 = local_gaussian_monte_carlo_bge_score(1, {0}, gaussian_data, interventions)
    s3 = local_gaussian_monte_carlo_bge_score(2, {0, 1}, gaussian_data, interventions)
    print("new result node 0", s1)
    print("new result node 1", s2)
    print("new result node 2", s3)
    print("total:", s1+s2+s3)

