import numpy as np
import neurolib.utils.adjoint as adj

np.set_printoptions(precision=8)

def generic_jacobian(state_, model, u_=None):
    N = model.params.N
    V = state_.shape[1]
    jac = np.zeros(( N,V,V ))
    dt = model.params.dt

    dx = 0.01

    init_vars = model.init_vars
    state_vars = model.state_vars

    #print(state_.shape)

    model.params.duration = 3. * dt
    adj.set_init(model, state_, init_vars, state_vars)
    model.run()
    T = np.around(model.params.duration/dt + 1,1).astype(int)
    x0 = adj.get_fullstate(model, state_vars, N, V, T)


    for n in range(N):
        for v in range(V):

            state1 = state_.copy()
            state1[n,v] += dx
            adj.set_init(model, state1, init_vars, state_vars)
            model.run()
            x1 = adj.get_fullstate(model, state_vars, N, V, T)

            #print(x0)
            #print(x1)

            for v_ in range(V):

                if v_ in adj.variables.algvar:
                    if v in adj.variables.algvar:
                        jac[n,v_,v] = ( x1[n,v_,0] - x0[n,v_,0] ) / dx
                    elif v in adj.variables.diffvar:
                        jac[n,v_,v] = ( x1[n,v_,1] - x0[n,v_,1] ) / dx
                elif v_ in adj.variables.diffvar:
                    dx0 = ( x0[n,v_,1] - x0[n,v_,0] ) / dt
                    dx1 = ( x1[n,v_,1] - x1[n,v_,0] ) / dt

                    jac[n,v_,v] = ( dx1 - dx0 ) / dx

    return jac

def generic_duh(state_, model, v_control, u_=None):
    N = model.params.N
    V = state_.shape[1]
    V_c = len(model.input_vars)
    duh = np.zeros(( N,V,V_c ))
    dt = model.params.dt

    du = 0.001

    init_vars = model.init_vars
    state_vars = model.state_vars

    model.params.duration = 3. * dt
    u0 = np.zeros(( N,V_c,4 ))
    adj.set_init(model, state_, init_vars, state_vars)
    adj.set_control(model, u0)
    model.run()
    T = np.around(model.params.duration/dt + 1,1).astype(int)
    x0 = adj.get_fullstate(model, state_vars, N, V, T)

    for n in range(N):
        for vc in v_control[n]:
            u1 = u0.copy()
            i0, i1 = 0, 1
            if model.name == 'aln':
                i0, i1 = 1, 0
            u1[n,vc,i0] += du
            adj.set_init(model, state_, init_vars, state_vars)
            adj.set_control(model, u1)
            #print('set control ', model.params.ext_exc_current)
            model.run()
            #print("tau_e = ", model.state['tau_exc'])
            x1 = adj.get_fullstate(model, state_vars, N, V, T)

            #print('mue 0', x0[0,2,:])
            #print(x1[0,2,:])

            for v in adj.variables.controlvar:

                dx0 = ( x0[n,v,i1] - x0[n,v,i0] ) / dt
                dx1 = ( x1[n,v,i1] - x1[n,v,i0] ) / dt

                duh[n,v,vc] = (dx1 - dx0) / du

    return duh

def jacobian_fhn(state_, model, u_=None):
    N = model.params.N
    V = state_.shape[1]
    jac = np.zeros(( N,V,V ))

    for n in range(N):

        jac[n,0,0] = - 3. * model.params.alpha * state_[n,0]**2 + 2. * model.params.beta * state_[n,0] + model.params.gamma
        jac[n,0,1] = -1.
        jac[n,1,0] = 1. / model.params.tau
        jac[n,1,1] = - model.params.epsilon / model.params.tau

    return jac

def jacobian_hopf(state_, model, u_=None):
    N = model.params.N
    V = state_.shape[1]
    jac = np.zeros(( N,V,V ))

    for n in range(N):

        jac[n,0,0] = model.params.a - 3. * state_[n,0]**2 - state_[n,1]**2
        jac[n,0,1] = - 2. * state_[n,0] * state_[n,1] - model.params.w
        jac[n,1,0] = - 2. * state_[n,0] * state_[n,1] + model.params.w
        jac[n,1,1] = model.params.a - state_[n,0]**2 - 3. * state_[n,1]**2

    return jac

def jacobian_wc(state_, model, u_):
    N = model.params.N
    V = state_.shape[1]
    jac = np.zeros(( N,V,V ))

    for n in range(N):

        exc_input = 0.

        expE = np.exp( - model.params.a_exc * ( ( model.params.c_excexc * state_[n,0] - model.params.c_inhexc * state_[n,1] + u_[n,0] ) - model.params.mu_exc  ))
        expI = np.exp( - model.params.a_inh * ( ( model.params.c_excinh * state_[n,0] - model.params.c_inhinh * state_[n,1] + u_[n,1] ) - model.params.mu_inh ) )
                
        jac[n,0,0] = ( - 1. - 1. / ( 1. + expE ) - ( 1. - state_[n,0] ) * ( - model.params.a_exc * model.params.c_excexc * expE) / ( 1. + expE )**2 ) / model.params.tau_exc
        jac[n,0,1] = ( ( state_[n,0] - 1. ) * ( model.params.a_exc * model.params.c_inhexc * expE) / ( 1. + expE )**2 ) / model.params.tau_exc
        jac[n,1,0] = ( ( state_[n,1] - 1. ) * ( - model.params.a_inh * model.params.c_excinh * expI) / ( 1. + expI )**2 ) / model.params.tau_inh
        jac[n,1,1] = ( - 1. - 1. / ( 1. + expI ) - ( 1. - state_[n,1] ) * ( model.params.a_inh * model.params.c_inhinh * expI) / ( 1. + expI )**2 ) / model.params.tau_inh

    return jac

def duh_fhn(state_, model, v_control, u_=None):
    N = model.params.N
    V = state_.shape[1]
    V_c = len(model.input_vars)
    duh = np.zeros(( N,V,V_c ))

    for n in range(N):
        if 0 in v_control[n]:
            duh[n,0,0] = 1.
        if 1 in v_control[n]:
            duh[n,1,1] = 1.        

    return duh

def duh_hopf(state_, model, v_control, u_=None):
    N = model.params.N
    V = state_.shape[1]
    V_c = len(model.input_vars)
    duh = np.zeros(( N,V,V_c ))

    for n in range(N):
        if 0 in v_control[n]:
            duh[n,0,0] = 1.
        if 1 in v_control[n]:
            duh[n,1,1] = 1.        

    return duh

def duh_wc(state_, model, v_control, u_):
    N = model.params.N
    V = state_.shape[1]
    V_c = len(model.input_vars)
    duh = np.zeros(( N,V,V_c ))

    for n in range(N):

        exc_input = 0.
        expE = np.exp( - model.params.a_exc * ( ( model.params.c_excexc * state_[n,0] - model.params.c_inhexc * state_[n,1] + exc_input + model.params.exc_ext_const[n] + u_[n,0] ) - model.params.mu_exc  ))
        expI = np.exp( - model.params.a_inh * ( ( model.params.c_excinh * state_[n,0] - model.params.c_inhinh * state_[n,1] + model.params.inh_ext_const[n] + u_[n,1] ) - model.params.mu_inh ) )

        if 0 in v_control[n]:
            duh[n,0,0] = ( ( state_[n,0] - 1. ) * ( - model.params.a_exc * expE) / ( 1. + expE )**2 ) / model.params.tau_exc
        if 1 in v_control[n]:
            duh[n,1,1] = ( ( state_[n,1] - 1. ) * ( - model.params.a_inh * expI) / ( 1. + expI )**2 ) / model.params.tau_inh

    return duh
