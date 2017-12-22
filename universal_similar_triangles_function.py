from math import sqrt
import numpy as np

def universal_similar_triangles_function(phi_big_oracle, phi_small_solver, primal_dual_oracle,
                                         t_start, L_init = 1.0, max_iter = 1000,
                                         epsilon = 1e-5, verbose = False):
    iter_step = 1
    L_value = L_init
    A_previous = 0.0
    A_current = None
    u_parameter = np.copy(t_start)
    t_parameter = np.copy(t_start)
    
    flows = np.zeros(len(t_start))
    y_parameter = np.copy(t_parameter)
    
    duality_gap_init = None
    epsilon_absolute = None
    
    for counter in range(0, max_iter):
        alpha = 0.5 / L_value + sqrt(0.25 / L_value**2 + A_previous / L_value)
        A_current = A_previous + alpha

        y_parameter = (alpha * u_parameter + A_previous * t_parameter) / A_current
        phi_small_solver.update(alpha, y_parameter)

        u_parameter = phi_small_solver.argmin_function(u_start = u_parameter)
        t_parameter = (alpha * u_parameter + A_previous * t_parameter) / A_current

        if counter == 0:
            duality_gap_init = primal_dual_oracle.duality_gap_function(t_parameter, - phi_big_oracle.grad(t_parameter))
            epsilon_absolute = epsilon * duality_gap_init
        
        left_value = (phi_big_oracle.func(y_parameter) + 
                      np.dot(phi_big_oracle.grad(y_parameter), t_parameter - y_parameter) + 
                      0.5 * alpha / A_current * epsilon_absolute) - phi_big_oracle.func(t_parameter)
        right_value = - 0.5 * L_value * np.sum(np.square(t_parameter - y_parameter))
        
        while (left_value < right_value):
            L_value = 2.0 * L_value
            right_value = 2.0 * right_value
            
        A_previous = A_current
        L_value = L_value / 2.0
        #if verbose:
        #    print('Iterations number: ' + str(counter + 1))
        
        if (counter + 1) % iter_step == 0:
            flows = - phi_big_oracle.grad(t_parameter)
            duality_gap = primal_dual_oracle.duality_gap_function(t_parameter, flows)

            if verbose:
                print('Iterations number: ' + str(counter + 1))
                print('Duality_gap / Duality_gap_init = ' + str(duality_gap / duality_gap_init))
                print('Duality_gap = ' + str(duality_gap))
            
            if duality_gap < epsilon_absolute:
                if verbose:
                    print('Success!  Iterations number: ' + str(counter + 1))
                return t_parameter, flows, counter, 'success'
                                
    return t_parameter, flows, counter, 'iterations_exceeded'

