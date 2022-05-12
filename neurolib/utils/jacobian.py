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

    model.params.duration = 3. * dt
    set_init(model, state_, init_vars, state_vars)
    model.run()
    T = np.around(model.params.duration/dt + 1,1).astype(int)
    x0 = get_fullstate(model, state_vars, N, V, T)

    for n in range(N):
        for v in range(V):
            state1 = state_.copy()
            state1[n,v] += dx
            set_init(model, state1, init_vars, state_vars)
            model.run()
            x1 = get_fullstate(model, state_vars, N, V, T)

            #print(x0)
            #print(x1)

            for v_ in range(V):

                dx0 = ( x0[n,v_,1] - x0[n,v_,0] ) / dt
                dx1 = ( x1[n,v_,1] - x1[n,v_,0] ) / dt

                jac[n,v_,v] = (dx1 - dx0) / dx

    """
    A_ = np.zeros((3,3))
    x_ = np.zeros(( 3 ))

    for t in range(3):
        A_[t,0] = x0[0,0,t]
        A_[t,1] = x0[0,1,t]
        A_[t,2] = 1.
        x_[t] = ( x0[0,0,t+1] - x0[0,0,t] ) / dt

    sol = np.linalg.solve(A_, x_)
    #print(sol)
    """

    return jac

def generic_duh(state_, model, v_control, u_=None):
    N = model.params.N
    V = state_.shape[1]
    V_c = len(model.input_vars)
    duh = np.zeros(( N,V,V_c ))
    dt = model.params.dt

    du = 0.01

    init_vars = model.init_vars
    state_vars = model.state_vars

    model.params.duration = 3. * dt
    u0 = np.zeros(( N,V_c,4 ))
    set_init(model, state_, init_vars, state_vars)
    adj.set_control(model, u0)
    model.run()
    T = np.around(model.params.duration/dt + 1,1).astype(int)
    x0 = get_fullstate(model, state_vars, N, V, T)

    for n in range(N):
        for vc in v_control[n]:
            u1 = u0.copy()
            u1[n,vc,0] += du
            set_init(model, state_, init_vars, state_vars)
            adj.set_control(model, u1)
            model.run()
            x1 = get_fullstate(model, state_vars, N, V, T)

            for v in range(V):

                dx0 = ( x0[n,v,1] - x0[n,v,0] ) / dt
                dx1 = ( x1[n,v,1] - x1[n,v,0] ) / dt

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

        expE = np.exp( - model.params.a_exc * ( ( model.params.c_excexc * state_[n,0] - model.params.c_inhexc * state_[n,1] + exc_input + model.params.exc_ext_const[n] + u_[n,0] ) - model.params.mu_exc  ))
        expI = np.exp( - model.params.a_inh * ( ( model.params.c_excinh * state_[n,0] - model.params.c_inhinh * state_[n,1] + model.params.inh_ext_const[n] + u_[n,1] ) - model.params.mu_inh ) )
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

def set_init(model, x0_, init_vars_, state_vars_):
    N = x0_.shape[0]
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
            x_[n,sv,:] = model[state_vars_[sv]][n,:]

    return x_
