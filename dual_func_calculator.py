import numpy as np

class PrimalDualCalculator:
    def __init__(self, phi_big_oracle, freeflowtimes, capacities, rho = 10.0, mu = 0.25):
        self.links_number = len(freeflowtimes)
        self.rho = rho
        self.mu = mu
        self.freeflowtimes = freeflowtimes #\bar{t}
        self.capacities = capacities       #\bar{f}
        
        self.phi_big_oracle = phi_big_oracle
    
    def dual_func_value(self, t_parameter):
        assert(not np.any(np.isnan(t_parameter)))
        h_function_value = np.sum(1.0 / (1.0 + self.mu) * self.capacities * \
                                  np.power(np.maximum(t_parameter - self.freeflowtimes, 0.0) / \
                                           (self.rho * self.freeflowtimes), self.mu) * \
                                  (t_parameter - self.freeflowtimes))
        return self.phi_big_oracle.func(t_parameter) + h_function_value
    
    
    def primal_func_value(self, flows, t_parameter):
        sigma_sum_function = np.sum(self.freeflowtimes * flows * (self.rho * self.mu / (1.0 + self.mu) * 
                                                          np.power(flows / self.capacities, 1.0 / self.mu) + 1.0))
        return sigma_sum_function - self.phi_big_oracle.entropy(t_parameter)
    
    def duality_gap_function(self, t_parameter, f_parameter):
        return self.primal_func_value(f_parameter, t_parameter) + self.dual_func_value(t_parameter)
               