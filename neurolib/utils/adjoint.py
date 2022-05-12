import numpy as np
import neurolib.utils.jacobian as jac

import matplotlib.pyplot as plt

np.set_printoptions(precision=8)

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
        
        for t in range(adjoint.shape[2]-2, -1, -1):
            jac_t1 = jacobian(st_[:,:,t+1], model, u0[:,:,t+1])
            for v in range(V):
                der = 0.
                der += fp[n,v,t+1]
                for v1 in range(V):
                    der += adjoint[n,v1,t+1] * jac_t1[n,v1,v]
                #print(t, fp[n,0,t+1], der)
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
        for n in range(N):
            for vc in range(V_c):
                grad[n,vc,t] = fu[n,vc,t]
                for v in range(V):
                    grad[n,vc,t] += adj_[n,v,t] * duh_[n,v,vc]
    return grad

def cost(model, state_, target_, w, u):
    cost = 0.
    N = model.params.N
    V = state_.shape[1]
    T = state_.shape[2]

    for n in range(N):
        for v in range(V):
            for t in range(T):
                cost += 0.5 * ( state_[n,v,t] - target_[n,v,t] )**2 + 0.5 * w * u[n,v,t]**2

    return cost

def bisection(model, dur_, init_state_, u0, d0, target_, w, T, maxcontrol, step0, factor):

    N = model.params.N
    V = init_state_.shape[1]

    state_vars = model.state_vars

    model.params.duration = dur_
    jac.set_init(model, init_state_, model.init_vars, state_vars)
    set_control(model, u0)
    model.run()
    state0 = jac.get_fullstate(model, model.state_vars, N, V, T)
    c0 = cost(model, state0, target_, w, u0)

    s = step0

    u1 = u0 + s * d0
    u1 = set_maxcontrol(u1, maxcontrol)
    set_control(model, u1)
    model.run()
    state1 = jac.get_fullstate(model, state_vars, N, V, T)
    c1 = cost(model, state1, target_, w, u1)


    u2 = u0 + factor * s * d0
    u2 = set_maxcontrol(u2, maxcontrol)
    set_control(model, u2)
    model.run()
    state2 = jac.get_fullstate(model, state_vars, N, V, T)
    c2 = cost(model, state2, target_, w, u2)

    while c2 <= c1:
        s *= factor
        
        u1 = u0 + s * d0
        u1 = set_maxcontrol(u1, maxcontrol)
        set_control(model, u1)
        model.run()
        state1 = jac.get_fullstate(model, state_vars, N, V, T)
        c1 = cost(model, state1, target_, w, u1)

        u2 = u0 + factor * s * d0
        u2 = set_maxcontrol(u2, maxcontrol)
        set_control(model, u2)
        model.run()
        state2 = jac.get_fullstate(model, state_vars, N, V, T)
        c2 = cost(model, state2, target_, w, u2)

        #print(s, c1, c2)

        if s < 1e-10:
            print("step size limit reached")
            return 0.
            
    #print(s, c1, c2)
    return s

def opt_c(model, max_it, init_state_, target_, w, u0, v_control, maxcontrol=10., method='generic', step0=100., factor=0.9): 

    cost_list = []

    N = model.params.N
    V = target_.shape[1]
    T = target_.shape[2]

    dt = model.params.dt
    dur_ = (T-1.)*dt

    zero_control = np.zeros(( u0.shape ))

    model.params.duration = dur_
    jac.set_init(model, init_state_, model.init_vars, model.state_vars)
    set_control(model, u0)
    model.run()
    state0 = jac.get_fullstate(model, model.state_vars, N, V, T)


    cost_list.append(cost(model, state0, target_, w, u0))
    print("Initial cost = ", cost_list[-1])

    for i in range(max_it):
        fp_ = fp(state0, target_)
        fu_ = fu(w, u0)
        adj = adjoint(model, state0, fp_, u0, method)
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
        jac.set_init(model, init_state_, model.init_vars, model.state_vars)
        set_control(model, u0)
        model.run()
        state0 = jac.get_fullstate(model, model.state_vars, N, V, T)

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
