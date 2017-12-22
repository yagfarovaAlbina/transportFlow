# from scipy.special import expit
#import multiprocessing as mp
from collections import defaultdict
from scipy.misc import logsumexp
from scipy.special import expit
import numpy as np


class BaseOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class AutomaticOracle(BaseOracle):
    """
    Oracle for automatic calculations of function kGamma * \Psi (t)
    """

    def __init__(self, source, graph, source_correspondences, gamma = 1.0):
        #may be it should be self.graph.kMaxPathLength
        self.graph = graph #should it be here? 
        self.source = self.graph.get_node_index(source)
        correspondences_targets = self.graph.get_nodes_indices(source_correspondences.keys())
        correspondences_values = np.array(list(source_correspondences.values()))
        nonzero_indices = np.nonzero(correspondences_values)
        self.correspondences_targets = correspondences_targets[nonzero_indices]
        self.correspondences_values = correspondences_values[nonzero_indices]
        self.gamma = gamma
        
        self.t_current = None
        
        #temp: for 
        self.log_entropy_array = None

    def func(self, t_parameter):
        #print('automatic func called...'+'t_parameter = ' + str(t_parameter) )
        self.t_current = t_parameter
        self._calculate_a_b_values()
        
        return np.dot(self.correspondences_values,
                      get_matrix_values(self.B_values, self.graph.kMaxPathLength,
                                        self.correspondences_targets))

    def grad(self, t_parameter):
        #assert(np.all(self.t_current == t_parameter))
        #print('automatic grad called...'+'t_parameter = ' + str(t_parameter) )
        gradient_vector = np.zeros(self.graph.kLinksNumber)
        
        #psi_d_beta_values initial values path_length = kMaxPathLength 
        psi_d_beta_values = np.zeros(self.graph.kNodesNumber)
        psi_d_beta_values[self.correspondences_targets] = self.correspondences_values
        psi_d_alpha_values = np.zeros(self.graph.kNodesNumber)
         
        alpha_d_time = np.zeros(self.graph.kLinksNumber)
        for edge in self.graph.edges():
            target_vertex = self.graph.target_of_edge(edge)
            alpha_d_time[edge] = self._alpha_d_time_function(self.graph.kMaxPathLength, target_vertex, edge)
        
        for path_length in range(self.graph.kMaxPathLength - 1, 0, -1):
            beta_d_beta_values = self._beta_d_beta_function(path_length + 1)
            psi_d_beta_values = psi_d_beta_values * beta_d_beta_values
            
            #calculating psi_d_alpha_values
            beta_d_alpha_values = self._beta_d_alpha_function(path_length)
            first_terms = psi_d_beta_values * beta_d_alpha_values
                
            second_terms = np.zeros(self.graph.kNodesNumber)
            for node in self.graph.nodes():
                if len(self.graph.successors(node)) > 0:
                    second_terms[node] = - np.dot(psi_d_alpha_values[self.graph.successors(node)],
                                                 alpha_d_time[self.graph.out_edges(node)])
                    #np.vectorize(graph.nodes())
            
            psi_d_alpha_values = first_terms + second_terms
            
            #calculating gradient
            for edge in self.graph.edges():
                target_vertex = self.graph.target_of_edge(edge)
                alpha_d_time[edge] = self._alpha_d_time_function(path_length, target_vertex, edge)
                gradient_vector[edge] += psi_d_alpha_values[target_vertex] * alpha_d_time[edge]
        #print('my result = ' + str(gradient_vector))
        return gradient_vector
        
    def entropy(self, t_parameter):
        #assert(np.all(self.t_current == t_parameter))
        log_entropy_array = - np.inf * np.ones(self.graph.kNodesNumber)
        
        prev_array = np.copy(log_entropy_array)
        current_array = np.empty(self.graph.kNodesNumber)
        with np.errstate(invalid='raise', divide='raise'):
            for path_length in range(2, self.graph.kMaxPathLength + 1):
                current_array.fill(- np.inf)
                for node in self.graph.nodes():
                    if not np.isinf(get_matrix_values(self.A_values, path_length, node)):
                        ent = []
                        for edge in self.graph.in_edges(node):
                            edge_source = self.graph.source_of_edge(edge)
                            if not np.isinf(get_matrix_values(self.A_values, path_length - 1, edge_source)):
                                coef = 1.0 / self.gamma * (get_matrix_values(self.A_values, path_length - 1, edge_source) - \
                                                           self.t_current[edge] - \
                                                           get_matrix_values(self.A_values, path_length, node))
                                ent.append(coef + prev_array[edge_source])
                                try:
                                    log_coef = np.log(- coef)
                                except FloatingPointError:
                                    log_coef = - np.inf
                                try:
                                    ent.append(coef + log_coef)
                                except Exception:
                                    print(coef, log_coef)

                        current_array[node] = logsumexp(ent)

                        if np.isinf(get_matrix_values(self.B_values, path_length - 1, node)):
                            log_entropy_array[node] = current_array[node]
                        else:
                            ent = []
                            coef_last = 1.0 / self.gamma * (get_matrix_values(self.B_values, path_length - 1, node) - \
                                                           get_matrix_values(self.B_values, path_length, node))
                            ent.append(coef_last + log_entropy_array[node])
                            try:
                                log_coef = np.log(- coef_last)
                            except FloatingPointError:
                                log_coef = - np.inf
                            ent.append(coef_last + log_coef)
                            coef_new = 1.0 / self.gamma * (get_matrix_values(self.A_values, path_length, node) - \
                                                           get_matrix_values(self.B_values, path_length, node))
                            ent.append(coef_new + current_array[node])
                            try:
                                log_coef = np.log(- coef_new)
                            except FloatingPointError:
                                log_coef = - np.inf
                            ent.append(coef_new + log_coef)
                            log_entropy_array[node] = logsumexp(ent)

                temp_array = prev_array
                prev_array = current_array
                current_array = temp_array
        result = np.exp(logsumexp(np.log(self.correspondences_values) + log_entropy_array[self.correspondences_targets]))
        self.log_entropy_array = log_entropy_array
        return result
    
    def time_av_w_array(self, t_parameter):
        return list(self.gamma * np.exp(self.log_entropy_array) - \
                    get_matrix_values(self.B_values, self.graph.kMaxPathLength))
    
    def _alpha_d_time_function(self, path_length, term_vertex, edge_index):
        #print('alpha_d_time_func called...')
        if self.graph.target_of_edge(edge_index) != term_vertex:
            return 0.0
        edge_source = self.graph.source_of_edge(edge_index)
        if path_length == 1:
            if  edge_source == self.source:
                return - 1.0
            else:
                return 0.0
        A_value_term = get_matrix_values(self.A_values, path_length, term_vertex)
        A_value_source = get_matrix_values(self.A_values, path_length - 1, edge_source)
        if np.isinf(A_value_term) or np.isinf(A_value_source):
            return 0.0
        return - np.exp(1.0 / self.gamma * (A_value_source - 
                                            self.t_current[edge_index] - A_value_term))

    
    def _alpha_d_alpha_functions(self, path_length, term_vertex, deriv_term_vertex, edge_index):
        #edge_index = self.graph.edge_index(deriv_term_vertex, term_vertex)
        result = - self._alpha_d_time_function(path_length, term_vertex, edge_index)
        return result

    def _beta_d_beta_function(self, path_length):
        if path_length == 1:
            return np.zeros(self.graph.kNodesNumber)
        alpha_values = get_matrix_values(self.A_values, path_length)
        beta_values = get_matrix_values(self.B_values, path_length - 1)
        
        indices = np.nonzero(np.logical_not(np.isinf(beta_values)))
        values = - np.inf * np.ones(self.graph.kNodesNumber)
        values[indices] = - alpha_values[indices] + beta_values[indices]
        #values = np.where(np.logical_and(np.isinf(alpha_values), np.isinf(beta_values)), -np.inf, alpha_values - beta_values)
        result = expit(values / self.gamma)
        return result


    def _beta_d_alpha_function(self, path_length):
        if path_length == 1:
            return np.ones(self.graph.kNodesNumber)
        alpha_values = get_matrix_values(self.A_values, path_length)
        beta_values = get_matrix_values(self.B_values, path_length - 1)
        
        indices = np.nonzero(np.logical_not(np.isinf(alpha_values)))
        values = - np.inf * np.ones(self.graph.kNodesNumber)
        values[indices] = alpha_values[indices] - beta_values[indices]
        #values = np.where(np.logical_and(np.isinf(alpha_values), np.isinf(beta_values)), -np.inf, alpha_values - beta_values)
        result = expit(values / self.gamma)
        return result

            
    def _calculate_a_b_values(self):
        self.A_values = - np.inf * np.ones((self.graph.kMaxPathLength, self.graph.kNodesNumber))
        self.B_values = - np.inf * np.ones((self.graph.kMaxPathLength, self.graph.kNodesNumber))
        initial_values = - 1.0 * self.t_current[self.graph.out_edges(self.source)]
        set_matrix_values(initial_values, self.A_values, 1, self.graph.successors(self.source))
        set_matrix_values(initial_values, self.B_values, 1, self.graph.successors(self.source))
        
        for path_length in range(2, self.graph.kMaxPathLength + 1):
            for term_vertex in self.graph.nodes():
                if len(self.graph.predecessors(term_vertex)) > 0:
                    alpha = self.gamma * \
                            logsumexp(1.0 / self.gamma * 
                                      (get_matrix_values(self.A_values, path_length - 1, self.graph.predecessors(term_vertex))
                                       - self.t_current[self.graph.in_edges(term_vertex)]))
                    
                    beta = self.gamma * \
                            logsumexp([1.0 / self.gamma * get_matrix_values(self.B_values, path_length - 1, term_vertex),
                                       1.0 / self.gamma * alpha])
                    
                    set_matrix_values(alpha, self.A_values, path_length, term_vertex)
                    set_matrix_values(beta, self.B_values, path_length, term_vertex)
                    
                    #print('path_length = ' + str(path_length) + ' term_vertex = ' + str(term_vertex))
                    #print(get_matrix_values(self.A_values, path_length - 1, self.graph.predecessors(term_vertex)))
                    #print(self.t_current[self.graph.in_edges(term_vertex)])
                    #print('a_value = ' + str(alpha))
                    #print('b_value = ' + str(beta))
         
        #assert(not np.any(np.isnan(self.A_values)))
        #assert(not np.any(np.isnan(self.B_values)))
        
        
def set_matrix_values(values, array, path_length, vertices_list = None):
    if vertices_list is not None:
        array[path_length - 1][np.array(vertices_list)] = np.array(values)
    else:
        array[path_length - 1][:] = np.array(values)

def get_matrix_values(array, path_length, vertices_list = None):
    if vertices_list is not None:
        res = array[path_length - 1][np.array(vertices_list)]
    else:
        res = array[path_length - 1][:]
    return res

def pickle_func(oracle, args):
    return oracle._process_func(*args)

class PhiBigOracle(BaseOracle):
    def __init__(self, graph, correspondences, processes_number = None, gamma = 1.0):
        self.graph = graph
        self.correspondences = correspondences
        if processes_number:
            self.processes_number = processes_number
        else:
            self.processes_number = len(correspondences)
        self.gamma = gamma
        self.t_current = None
        self.func_current = None
        self.grad_current = None
        
        self.auto_oracles = []
        for source, source_correspondences in self.correspondences.items():
            self.auto_oracles.append(AutomaticOracle(source, self.graph, source_correspondences, gamma = self.gamma))
            
    def _reset(self, t_parameter):
        self.t_current = t_parameter
        self.func_current = 0.0
        self.grad_current = np.zeros(self.graph.kLinksNumber)
        for auto_oracle in self.auto_oracles:
            self.func_current += auto_oracle.func(self.t_current)
            self.grad_current += auto_oracle.grad(self.t_current)
    
    def func(self, t_parameter):
        if self.t_current is None or np.any(self.t_current != t_parameter):
            self._reset(t_parameter)
        return self.func_current
            
    def grad(self, t_parameter):
        if self.t_current is None or np.any(self.t_current != t_parameter):
            self._reset(t_parameter)
        return self.grad_current
    
    def entropy(self, t_parameter):
        if self.t_current is None or np.any(self.t_current != t_parameter):
            self._reset(t_parameter)
        return self.gamma * np.sum([auto_oracle.entropy(t_parameter) for auto_oracle in self.auto_oracles])
    
    def time_av_w_matrix(self, t_parameter):
        if self.t_current is None or np.any(self.t_current != t_parameter):
            self._reset(t_parameter)
        return np.array([auto_oracle.time_av_w_array(t_parameter) for auto_oracle in self.auto_oracles])
            
"""    
    def _process_func(self, source, graph,
                      source_correspondences, operation, t_parameter):
        automatic_oracle = AutomaticOracle(source, graph, source_correspondences)
        if operation == 'func':
            res = automatic_oracle.func(t_parameter)
        if operation == 'grad':
            res = automatic_oracle.grad(t_parameter)
        return res
                        
    def func(self, t_parameter):
        if self.t_current is None or np.any(self.t_current != t_parameter):
            self.t_current = t_parameter
            #pool = mp.Pool(processes = self.processes_number)
            results = []
            for key, value in self.correspondences.iteritems():
                #results.append(pool.apply_async(pickle_func, args=(self, (key, self.graph, value, 'func', t_parameter))))
                results.append(pickle_func(self, (key, self.graph, value, 'func', t_parameter)))
            #results = np.array([p.get() for p in results])
            results = np.array(results)
            self.func_current = np.sum(results)
            pool.close()
        return self.func_current
    
    def grad(self, t_parameter):
        if self.t_current is None or np.any(self.t_current != t_parameter):
            pool = mp.Pool(processes = self.processes_number)
            results = []
            for key, value in self.correspondences.iteritems():
                #results.append(pool.apply_async(pickle_func, args=(self, (key, self.graph, value, 'grad', t_parameter))))
                results.append(pickle_func(self, (key, self.graph, value, 'grad', t_parameter)))
            #results = np.array([p.get() for p in results])
            results = np.array(results)
            self.grad_current = np.sum(results, axis = 0)
            pool.close()
        return self.grad_current
"""

