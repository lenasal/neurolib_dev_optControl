import logging
import numpy as np
import numba
from numba.typed import List

from ..utils.collections import dotdict

costparams = dotdict({})
costparamsdefault = np.array([1., 1., 1., 1.])
tolerance = 1e-16

#@numba.njit
def makeList(list_):
    l = List()
    for l0 in list_:
        l.append(l0)
    return l

def getParams():
    if (len(costparams) == 0):
        logging.warn("Cost parameters not found, set default values.")
        setDefaultParams()
    return costparams.I_p, costparams.I_e, costparams.I_s

def getParam_r():
    return costparams.I_ra

def setParams(I_p, I_e, I_s, I_ra = 0.):
    print("set cost params: ", I_p, I_e, I_s, I_ra)
    if (I_p < 0):
        logging.error("Cost parameter I_p smaller 0 not allowed, use default instead")
        costparams.I_p = costparamsdefault[0]
    else:
        costparams.I_p = I_p
    if (I_e < 0):
        logging.error("Cost parameter I_e smaller 0 not allowed, use default instead")
        costparams.I_e = costparamsdefault[1]
    else:
        costparams.I_e = I_e
    if (I_s < 0):
        logging.error("Cost parameter I_s smaller 0 not allowed, use default instead")
        costparams.I_s = costparamsdefault[2]
    else:
        costparams.I_s = I_s
    if (I_ra < 0):
        logging.error("Cost parameter I_ra smaller 0 not allowed, use default instead")
        costparams.I_ra = costparamsdefault[3]
    else:
        costparams.I_ra = I_ra

def setDefaultParams():
    print("set default params")
    #costparams = dotdict({})
    costparams.I_p = costparamsdefault[0]
    costparams.I_e = costparamsdefault[1]
    costparams.I_s = costparamsdefault[2]

    costparams.I_ra = costparamsdefault[3]

###########################################################
# cost functions for precision
###########################################################

# gradient of cost function for precision at time t
# time interval for transition can be set by defining respective target state to -1.
def cost_precision_gradient_t(N, V_target, state_t_, target_state_t_, i_p):
    cost_gradient_ = numba_precision_gradient_t(N, V_target, i_p, state_t_, target_state_t_)
    return cost_gradient_

@numba.njit
def numba_precision_gradient_t(N, V_target, i_p, state_t_, target_state_t_):
    cost_gradient_ = np.zeros(( N, V_target ))
    for ind_node in range(N):
        for ind_var in range(V_target):
            if target_state_t_[ind_node, ind_var] == -1000:
                cost_gradient_[ind_node, ind_var] += 0.
            else:
                cost_gradient_[ind_node, ind_var] += i_p * (state_t_[ind_node, ind_var] - 
                                               target_state_t_[ind_node, ind_var])
    return cost_gradient_ 

def cost_precision_int(N, T, dt, i_p, state_, target_, va_):
    if N == 1:
        return numba_cost_precision_int(N, T, dt, i_p, state_, target_, var_ = va_ )
    else:
        return numba_cost_precision_int_network(N, T, dt, i_p, state_, target_, var_ = va_ )

@numba.njit
def numba_cost_precision_int(N, T, dt, i_p, state_, target_state_, var_):
    cost =  0.
    t0 = 0.
    for ind_time in range(T):
        for ind_node in range(N):
            for ind_var in var_:
                diff = np.abs(state_[ind_node, ind_var, ind_time] - target_state_[ind_node, ind_var, ind_time])
                if target_state_[ind_node, ind_var, ind_time] == -1000:
                    cost += 0.
                    t0 = ind_time
                elif ( diff < tolerance ):
                    cost += 0.
                else:
                    cost += dt * 0.5 * i_p * diff**2.

    return cost

@numba.njit
def numba_cost_precision_int_network(N, T, dt, i_p, state_, target_state_, var_):
    cost =  0.
    t0 = 0.
    for ind_time in range(T):
        for ind_node in range(N):
            if var_[ind_node][0] == 1 and var_[ind_node][1] == 1:
                var_list = np.array([0,1])
            elif var_[ind_node][0] == 1:
                var_list = np.array([0])
            elif var_[ind_node][1] == 1:
                var_list = np.array([1])
            else:
                continue

            for ind_var in var_list:
                diff = np.abs(state_[ind_node, ind_var, ind_time] - target_state_[ind_node, ind_var, ind_time])
                if target_state_[ind_node, ind_var, ind_time] == -1000:
                    cost += 0.
                    t0 = ind_time
                elif ( diff < tolerance ):
                    cost += 0.
                else:
                    cost += dt * 0.5 * i_p * diff**2.

    return cost

def cost_precision_node(N, T, dt, i_p, state_, target_, va_):
    var = makeList(va_)
    if N == 1:
        return numba_cost_precision_node(N, T, dt, i_p, state_, target_, var_ = var )
    else:
        return numba_cost_precision_node_network(N, T, dt, i_p, state_, target_, var_ = var )

@numba.njit
def numba_cost_precision_node(N, T, dt, i_p, state_, target_state_, var_):
    cost =  np.zeros(( N, 2 ))
    t0 = 0.

    var_list = var_

    for ind_node in range(N):
        for ind_var in var_:
            for ind_time in range(T):
                if target_state_[ind_node, ind_var, ind_time] == -1000:
                    cost[ind_node, ind_var] += 0.
                    t0 = ind_time
                else:
                    diff = np.abs(state_[ind_node, ind_var, ind_time] - target_state_[ind_node, ind_var, ind_time])
                    if ( diff < tolerance ):
                        cost[ind_node, ind_var] += 0.
                    else:
                        cost[ind_node, ind_var] += dt * 0.5 * i_p * diff**2.
    return cost

@numba.njit
def numba_cost_precision_node_network(N, T, dt, i_p, state_, target_state_, var_):
    cost =  np.zeros(( N, 2 ))
    t0 = 0.
     
    for ind_node in range(N):

        if var_[ind_node][0] == 1 and var_[ind_node][1] == 1:
            var_list = np.array([0,1])
        elif var_[ind_node][0] == 1:
            var_list = np.array([0])
        elif var_[ind_node][1] == 1:
            var_list = np.array([1])
        else:
            var_list = np.array([0,1])

        for ind_var in var_list:
            for ind_time in range(T):
                if target_state_[ind_node, ind_var, ind_time] == -1000:
                    cost[ind_node, ind_var] += 0.
                    t0 = ind_time
                else:
                    diff = np.abs(state_[ind_node, ind_var, ind_time] - target_state_[ind_node, ind_var, ind_time])
                    if ( diff < tolerance ):
                        cost[ind_node, ind_var] += 0.
                    else:
                        cost[ind_node, ind_var] += dt * 0.5 * i_p * diff**2.
    return cost 
    
###########################################################
# cost functions for energy
###########################################################    

# gradient of cost function for energy
def cost_energy_gradient(control_, i_e):
    # state_t: [N,dim_Model] dimensional array containing all nodes and state variables for all times
    cost_gradient_e_ = numba_energy_gradient(i_e, control_)
    return cost_gradient_e_

@numba.njit
def numba_energy_gradient(i_e, control_):
    cost_grad_ = i_e * control_.copy()
    return cost_grad_

def cost_energy_int(N, V, T, dt, i_e, control_):
    cost_ = numba_cost_energy_int(N, V, T, dt, i_e, control_)
    return cost_

@numba.njit
def numba_cost_energy_int(N, V, T, dt, i_e, control_):
    cost =  0.
    for ind_time in range(T):
        for ind_node in range(N):
            for ind_var in range(V):
                cost += dt * 0.5 * i_e * control_[ind_node, ind_var, ind_time]**2
    return cost

def cost_energy_node(N, V, T, dt, i_e, control_, va_ = [0,1]):
    cost_ = numba_cost_energy_node(N, V, T, dt, i_e, control_)
    return cost_

@numba.njit
def numba_cost_energy_node(N, V, T, dt, i_e, control_):
    cost =  np.zeros(( N, V ))
    for ind_node in range(N):
        for ind_var in range(V):
            for ind_time in range(T):
                cost[ind_node,ind_var] += dt * 0.5 * i_e * control_[ind_node, ind_var, ind_time]**2
    return cost

@numba.njit
def control_energy_components(N, V, T, dt, control_):
    control_energy = np.zeros(( N, V ))
    for ind_node in range(N):
        for ind_var in range(V):
            energy = 0.
            for ind_time in range(0,T):
                energy += dt * control_[ind_node, ind_var, ind_time]**2
            control_energy[ind_node, ind_var] = np.sqrt(energy)
    return control_energy

###########################################################
# cost functions for sparsity
########################################################### 

# cost function for sparsity: simple absolute value
def cost_sparsity_gradient(N, V, T, dt, control_, i_s):
    control_energy = control_energy_components(N, V, T, dt, control_)
    cost_gradient_s_ =  numba_cost_sparsity_gradient(N, V, T, i_s, control_, control_energy)
    return cost_gradient_s_

@numba.njit
def numba_cost_sparsity_gradient(N, V, T, i_s, control_, control_energy):
    cost_grad =  np.zeros(( N, V, T ))
    
    if i_s != 0.:
        for ind_node in range(N):
            for ind_var in range(V):
                if control_energy[ind_node, ind_var] == 0.:
                    cost_grad[ind_node, ind_var, :] = 0.
                else:
                    cost_grad[ind_node, ind_var, :] = i_s * control_[ind_node, ind_var,:] / control_energy[ind_node, ind_var]
        #cost_grad[ind_node, ind_var, 0] = 0.
        
    return cost_grad

def f_cost_sparsity_int(N, V, T, dt, i_s, control_):
    cost =  numba_cost_sparsity_int(N, V, T, i_s, dt, control_)
    return cost

@numba.njit
def numba_cost_sparsity_int(N, V, T, i_s, dt, control_):
    int_ =  0.
    for ind_node in range(N):
        for ind_var in range(V):
            cost = 0.
            for ind_time in range(0,T):
                cost += (control_[ind_node, ind_var, ind_time])**2 * dt
            int_ += i_s * np.sqrt(cost)
    return int_

def f_cost_sparsity_node(N, V, T, dt, i_s, control_):
    cost =  numba_cost_sparsity_node(N, V, T, i_s, dt, control_)
    return cost

@numba.njit
def numba_cost_sparsity_node(N, V, T, i_s, dt, control_):
    int_ =  np.zeros(( N, V ))
    for ind_var in range(V):
        for ind_node in range(N):
            cost = 0.
            for ind_time in range(0,T):
                cost += (control_[ind_node, ind_var, ind_time])**2 * dt
            int_[ind_node, ind_var] += i_s * np.sqrt(cost)
    return int_

###########################################################
# cost functions for reproducibility auto-correlation
########################################################### 
def cost_ra_gradient_i(N, V, T, dt, i_ra, control_array, i):
    cost = numba_cost_ra_gradient(N, V, T, dt, i_ra, control_array, i)
    return cost

@numba.njit
def numba_cost_ra_gradient(N_noise, N, V, T, dt, i_ra, control_array, i):
    mean_control_ = numba_mean_control(N_noise, N, V, T, control_array)
    sigma_sq_control_ = numba_sigma_sq_control(N_noise, N, V, T, control_array, mean_control_)
    grad = np.zeros(( mean_control_.shape ))
    for ind_node in range(N):
            for ind_var in range(V):
                for ind_time in range(0,T):
                    grad[ind_node, ind_var, ind_time] += i_ra * (control_array[i,ind_node, ind_var, ind_time - mean_control_[ind_node, ind_var, ind_time]]) / sigma_sq_control_[ind_node, ind_var, ind_time]
    return grad

def cost_ra_int(N, V, T, dt, i_ra, control_array):
    cost = numba_cost_ra_int(N, V, T, dt, i_ra, control_array)
    return cost

@numba.njit
def numba_cost_ra_int(N_noise, N, V, T, dt, i_ra, control_array):
    mean_control_ = numba_mean_control(N_noise, N, V, T, control_array)
    sigma_sq_control_ = numba_sigma_sq_control(N_noise, N, V, T, control_array, mean_control_)
    cost = 0.
    for n in N_noise:
        cost += numba_cost_ra_i_int(N, V, T, dt, i_ra, control_array[n,:,:,:], mean_control_, sigma_sq_control_)
    return cost


@numba.njit
def numba_cost_ra_i_int(N, V, T, dt, i_ra, control_, mean_control_, sigma_sq_control_):
    cost = 0.

    for ind_node in range(N):
        for ind_var in range(V):
            for ind_time in range(0,T):
                cost += (control_[ind_node, ind_var, ind_time - mean_control_[ind_node, ind_var, ind_time]])**2 * dt / sigma_sq_control_[ind_node, ind_var, ind_time]
    cost += i_ra
    return cost

@numba.njit
def numba_mean_control(N_noise, N, V, T, control_array):
    mean_ = np.zeros((control_array[0].shape))
    for ind_node in range(N):
        for ind_var in range(V):
            for ind_time in range(0,T):
                for n in N_noise:
                    mean_[ind_node, ind_var, ind_time] += control_array[n,ind_node, ind_var, ind_time]

                mean_[ind_node, ind_var, ind_time] /= N

    return mean_

@numba.njit
def numba_sigma_sq_control(N_noise, N, V, T, control_array, mean_control_):
    sigma_sq_ = np.zeros((control_array[0].shape))
    for ind_node in range(N):
        for ind_var in range(V):
            for ind_time in range(0,T):
                for n in N_noise:
                    sigma_sq_[ind_node, ind_var, ind_time] += ( control_array[n,ind_node, ind_var, ind_time] - mean_control_[ind_node, ind_var, ind_time] )**2

                sigma_sq_[ind_node, ind_var, ind_time] /= N

    return sigma_sq_

# integrated cost
#@numba.njit
def f_int(N, V, T, dt, state_, target_, control_, i_p, i_e, i_s, v_ = [0,1]):
    # cost_: [t] dimensional array containing cost for all times
    # return cost_int: integrated (total) cost

    if N < 2:        
        var = makeList(v_)
    else:
        var = v_
            
    cost_prec, cost_energy, cost_sparsity = 0., 0., 0.
            
    if not i_p < 1e-12:
        cost_prec = cost_precision_int(N, T, dt, i_p, state_, target_, va_ = var)
    if not i_e < 1e-12:
        cost_energy = cost_energy_int(N, V, T, dt, i_e, control_)
    if not i_s < 1e-12:
        cost_sparsity = f_cost_sparsity_int(N, V, T, dt, i_s, control_)
        
    cost_int = cost_prec + cost_energy + cost_sparsity

    #print('cost splitting   : ', cost_prec, cost_energy, cost_sparsity)
    
    return cost_int

def cost_int_per_node(N, V, T, dt, state_, target_, control_, i_p, i_e, i_s, v_ = [0,1]):
    # cost_: [t] dimensional array containing cost for all times
    # return cost_int: integrated (total) cost
        
    var = makeList(v_)
        
    cost_prec_node = np.zeros(( N, 2 ))
    cost_e_node = np.zeros(( N, V ))
    cost_s_node = np.zeros(( N, V ))
            
    if not i_p < 1e-12:
        cost_prec_node = cost_precision_node(N, T, dt, i_p, state_, target_, va_ = var)
    if not i_e < 1e-12:
        cost_e_node = cost_energy_node(N, V, T, dt, i_e, control_)
    if not i_s < 1e-12:
        cost_s_node = f_cost_sparsity_node(N, V, T, dt, i_s, control_)
    
    """
    print("cost precision = ", cost_prec)
    print("cost energy = ", cost_energy)
    print("cost sparsity = ", cost_sparsity)
    """
    
    return [cost_prec_node, cost_e_node, cost_s_node]