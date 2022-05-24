import numpy as np
import neurolib.utils.jacobian as jac

import matplotlib.pyplot as plt
from ..utils.collections import dotdict

variables = dotdict({})

np.set_printoptions(precision=8)

def set_vartype(model):
    if model.name in ['fhn', 'hopf', 'wc']:
        variables.algvar = []
        variables.diffvar = np.arange(len(model.state_vars), dtype=int)
        variables.controlvar = [0,1]
    elif model.name == 'aln':
        variables.algvar = [0,1,15,16,18,19]
        variables.diffvar = [2,3,4,5,6,7,8,9,10,11,12,13,14,17]
        variables.controlvar = [2,3,5,6,7,8]
    else:
        return

def adjoint(model, st_, fp, u0, method='generic'):

    if method == 'generic':
        jacobian = jac.generic_jacobian
    elif method == 'analytic':
        if model.name == 'fhn':
            jacobian = jac.jacobian_fhn
        elif model.name == 'hopf':
            jacobian = jac.jacobian_hopf
        elif model.name == 'wc':
            jacobian = jac.jacobian_wc
        else:
            print("Analytic jacobian not implemented for {} model.".format(model.name))
            return None
    else:
        print("ERROR: no valid method for jacobian.")
        return None
    
    adjoint = np.zeros((st_.shape))
    #print(adjoint.shape)

    N = model.params.N
    V = st_.shape[1]
    dt = model.params.dt

    for n in range(N):

        t = adjoint.shape[2]-1
        jac_t0 = jacobian(st_[:,:,t], model, u0[:,:,t])
        
        for v in variables.algvar:
            adjoint[n,v,t] = 0.
            if v < fp.shape[1]:
                adjoint[n,v,t] -= fp[n,v,t]

        #for v in variables.diffvar:
        #    der = 0.
        #    for v1 in variables.algvar:
        #        der += adjoint[n,v1,t] * jac_t0[n,v1,v]
        #    adjoint[n,v,t] = dt * der

        
        for t in range(adjoint.shape[2]-2, -1, -1):

            jac_t1 = jacobian(st_[:,:,t+1], model, u0[:,:,t+1])
            jac_t0 = jacobian(st_[:,:,t], model, u0[:,:,t])

            for v in variables.algvar:
                adjoint[n,v,t] = 0.
                if v < fp.shape[1]:
                    adjoint[n,v,t] -= fp[n,v,t]
                for v1 in range(V):
                    if v1 == v:
                        continue
                    adjoint[n,v,t] -= adjoint[n,v1,t+1] * jac_t1[n,v1,v]

            for v in variables.diffvar:
                der = 0.
                if v < fp.shape[1]:
                    der += fp[n,v,t+1]
                for v1 in variables.diffvar:
                    der += adjoint[n,v1,t+1] * jac_t1[n,v1,v]
                for v1 in variables.algvar:
                    der += adjoint[n,v1,t+1] * jac_t1[n,v1,v]
                adjoint[n,v,t] = adjoint[n,v,t+1] + dt * der
            

                                
    return adjoint

def fp(state_, target_):
    return (state_ - target_)

def fu(w, u):
    return w*u
    
def grad(adj_, state_, fu, u0, v_control, model, method='generic'):

    if method == 'generic':
        duh = jac.generic_duh
    elif method == 'analytic':
        if model.name == 'fhn':
            duh = jac.duh_fhn
        elif model.name == 'hopf':
            duh = jac.duh_hopf
        elif model.name == 'wc':
            duh = jac.duh_wc
        else:
            print("Analytic duh not implemented for {} model.".format(model.name))
            return None
    else:
        print("ERROR: no valid method for duh.")
        return None

    N = model.params.N
    V = adj_.shape[1]
    T = adj_.shape[2]
    V_c = len(model.input_vars)

    grad = np.zeros(( N, V_c, T ))

    for t in range(T):  
        duh_ = duh(state_[:,:,t], model, v_control, u0[:,:,t])

        #print('duh 2,0 ', duh_[0,2,0])
        for n in range(N):
            for vc in range(V_c):
                grad[n,vc,t] = fu[n,vc,t]
                for v in range(V):
                    grad[n,vc,t] += adj_[n,v,t] * duh_[n,v,vc]

    #if model.name == 'aln':
    #    grad[:,:,1:] = grad[:,:,:-1]
    #    grad[:,:,0] = 0.
    return grad

def cost(model, state_, target_, w, u):
    cost = 0.
    N = model.params.N
    V_target = target_.shape[1]
    V_control = u.shape[1]
    T = state_.shape[2]

    for n in range(N):
        for t in range(T):
            for v in range(V_target):
                cost += 0.5 * ( state_[n,v,t] - target_[n,v,t] )**2
            for v in range(V_control):
                cost += 0.5 * w * u[n,v,t]**2

    return cost

def bisection(model, dur_, init_state_, u0, d0, target_, w, T, maxcontrol, step0, factor):

    N = model.params.N
    V = init_state_.shape[1]

    state_vars = model.state_vars

    model.params.duration = dur_
    set_init(model, init_state_, model.init_vars, state_vars)
    set_control(model, u0)
    model.run()
    state0 = get_fullstate(model, model.state_vars, N, V, T)
    c0 = cost(model, state0, target_, w, u0)

    s = step0

    u1 = u0 + s * d0
    u1 = set_maxcontrol(u1, maxcontrol)
    set_control(model, u1)
    model.run()
    state1 = get_fullstate(model, state_vars, N, V, T)
    c1 = cost(model, state1, target_, w, u1)


    u2 = u0 + factor * s * d0
    u2 = set_maxcontrol(u2, maxcontrol)
    set_control(model, u2)
    model.run()
    state2 = get_fullstate(model, state_vars, N, V, T)
    c2 = cost(model, state2, target_, w, u2)

    #print(u0[0,0,:], c0)
    #print(u1[0,0,:], c1)
    #print(u2[0,0,:], c2)

    #print(c0, c1, c2)

    while c2 <= c1 or c1 > c0:
        s *= factor
        
        u1 = u0 + s * d0
        u1 = set_maxcontrol(u1, maxcontrol)
        set_control(model, u1)
        model.run()
        state1 = get_fullstate(model, state_vars, N, V, T)
        c1 = cost(model, state1, target_, w, u1)

        u2 = u0 + factor * s * d0
        u2 = set_maxcontrol(u2, maxcontrol)
        set_control(model, u2)
        model.run()
        state2 = get_fullstate(model, state_vars, N, V, T)
        c2 = cost(model, state2, target_, w, u2)

        #print(s, c1, c2)

        if s < 1e-10:
            print("step size limit reached")
            return 0.
            
    #print(s, c1, c2)
    return s

def opt_c(model, max_it, init_state_, target_, w, u0, v_control, maxcontrol=10., method='generic', step0=10., factor=0.9): 

    set_vartype(model)
    cost_list = []

    N = model.params.N
    V_target = target_.shape[1]
    V_state = init_state_.shape[1]
    T = target_.shape[2]

    dt = model.params.dt
    dur_ = (T-1.)*dt

    zero_control = np.zeros(( u0.shape ))

    model.params.duration = dur_
    set_init(model, init_state_, model.init_vars, model.state_vars)
    set_control(model, u0)
    model.run()
    state0 = get_fullstate(model, model.state_vars, N, V_state, T)
    state0_targetvars = get_fullstate(model, model.state_vars, N, V_target, T)


    cost_list.append(cost(model, state0, target_, w, u0))
    print("Initial cost = ", cost_list[-1])

    for i in range(max_it):
        fp_ = fp(state0_targetvars, target_)
        #print('fp = ', fp_[0,0,:])
        fu_ = fu(w, u0)
        #print(state0.shape)
        adj = adjoint(model, state0, fp_, u0, method)
        #print("adj 0 = ", adj[0,0,:])
        #print("adj = ", adj[0,2,:])
        direction = - grad(adj, state0, fu_, u0, v_control, model, method)
        #print('direction ', direction[0,0,:])
        #print('direction ', direction[0,1,:])
        step = bisection(model, dur_, init_state_, u0, direction, target_, w, T, maxcontrol, step0, factor)
        if step == 0.:
            print("iteration ", i)
            break
        u0 = u0 + step * direction
        u0 = set_maxcontrol(u0, maxcontrol)
        #print('control ', u0[0,0,:])

        model.params.duration = dur_
        set_init(model, init_state_, model.init_vars, model.state_vars)
        set_control(model, u0)
        model.run()
        state0 = get_fullstate(model, model.state_vars, N, V_state, T)
        state0_targetvars = get_fullstate(model, model.state_vars, N, V_target, T)

        cost_list.append(cost(model, state0, target_, w, u0))

        if i < 10 or (i+1)%50 == 0 or i == max_it-1:
            print(i+1, " cost = ", cost_list[-1])

    return u0, cost_list

def set_control(model, control_):
    input_vars = model.input_vars

    for cv in range(len(input_vars)):
        c_cv = control_[:,cv,:] 
        model.params[input_vars[cv]] = c_cv


def set_maxcontrol(u0, maxcontrol):
    for n in range(u0.shape[0]):
        for v in range(u0.shape[1]):
            for t in range(u0.shape[2]):
                if u0[n,v,t] > maxcontrol:
                    u0[n,v,t] = maxcontrol
                elif u0[n,v,t] < - maxcontrol:
                    u0[n,v,t] = -maxcontrol

    return u0

def set_init(model, x0_, init_vars_, state_vars_):
    N = x0_.shape[0]
    #print(x0_.shape)
    #print(init_vars_)
    #print(state_vars_)
    for iv in range(len(init_vars_)):
        if 'ou' in init_vars_[iv]:
            continue
        for sv in range(len(state_vars_)):
            if state_vars_[sv] in init_vars_[iv]:
                init_array = np.zeros(( N,1 ))
                init_array[:,0] = x0_[:,sv]
                model.params[init_vars_[iv]] = init_array

def get_fullstate(model, state_vars_, N, V, T):
    x_ = np.zeros(( N,V,T ))
    for sv in range(V):
        if 'ou' in state_vars_[sv]:
            continue
        for n in range(N):
            #print(sv, state_vars_[sv], model.state[state_vars_[sv]])
            if len(model.state[state_vars_[sv]].shape) == 2:  # time series of state variable
                x_[n,sv,:] = model.state[state_vars_[sv]][n,:]
            else:
                x_[n,sv,:] = model.state[state_vars_[sv]][n]

    return x_
