import numpy as np
import logging
import numba
from . import costFunctions as cost

def updateState(model, control_):
    # set initial conditions once in other function
    state_ = model.getZeroState()
    output_vars = model.output_vars
    model.run(control = control_)    
    for i in range(len(output_vars)):
        state_[:,i,:] = model[output_vars[i]][:,:]
    #print("update state: ", state_[0,:,:3])
    return state_

def updateFullState(model, control_, state_vars_):
    state_ = model.getZeroFullState()
    model.run(control=control_)
    for sv in range(len(state_vars_)):
        state_[:,sv,:] = model.state[state_vars_[sv]][:,:]
    return state_

def get_output_from_full(model, state_):
    out_state_ = model.getZeroState()
    out_state_[:,0,:] = state_[:,0,:]
    out_state_[:,1,:] = state_[:,1,:]
    out_state_[:,2,:] = state_[:,4,:]
    return out_state_

def get_full_from_output_t(model, out_grad_):
    full_grad_ = np.zeros((model.params.N, len(model.state_vars) ))
    full_grad_[:,0] = out_grad_[:,0]
    full_grad_[:,1] = out_grad_[:,1]
    return full_grad_

def get_full_from_output(model, out_):
    full_ = model.getZeroFullState()
    full_[:,0,:] = out_[:,0,:]
    full_[:,1] = out_[:,1,:]
    return full_

def update_delayed_state(model, delay_state_vars_, state_vars_, init_vars_, startind_):
    for sv, iv in zip( range(len(state_vars_)), range(len(init_vars_)) ):
        if (state_vars_[sv] in init_vars_[iv]):
            if model.params[init_vars_[iv]].ndim == 2:
                if startind_ == 1:
                    model.params[init_vars_[iv]][:,0] = model.state[state_vars_[sv]][:,-2]
                else:
                    delay_state_vars_[:, sv, :-1] = delay_state_vars_[:, sv, 1:]
                    delay_state_vars_[:, sv, -1] = model.state[state_vars_[sv]][:,-2]
                    model.params[init_vars_[iv]][:,:] = delay_state_vars_[:, sv, :]
            else:
                if model.state[state_vars_[sv]].ndim == 2:
                    model.params[init_vars_[iv]] = model.state[state_vars_[sv]][:,-2]
                else:
                    model.params[init_vars_[iv]] = model.state[state_vars_[sv]]
        else:
            logging.error("Initial and state variable labelling does not agree.")

def adjust_shape_init_params(model, init_vars_, startind_):
    if model.params.N == 1:
        for iv in range(len(init_vars_)):
            if model.params[init_vars_[iv]].ndim == 2:
                if (model.params[init_vars_[iv]].shape[1] <= 1):
                    model.params[init_vars_[iv]] = np.dot( model.params[init_vars_[iv]], np.ones((1, startind_)) )
            else:
                model.params[init_vars_[iv]] = model.params[init_vars_[iv]][0] * np.ones((1, startind_))
    else:
        for iv in range(len(init_vars_)):
            initvars = np.zeros(( model.params.N, startind_ ))
            for n in range(model.params.N):
                initvars[n,:] = model.params[init_vars_[iv]][n]

            model.params[init_vars_[iv]] = initvars

def get_init(model, init_vars_, state_vars_, startind_, delay_state_vars_):
    IC_ = np.zeros( (model.params.N, len(init_vars_), startind_) )
    
    for iv in range(len(init_vars_)):
        if ( type(model.params[init_vars_[iv]]) == np.float64 or type(model.params[init_vars_[iv]]) == float ):
            IC_[:, iv, 0] = model.params[init_vars_[iv]]
        elif len(model.params[init_vars_[iv]][:].shape) == 2:
            if startind_ == 1:
                IC_[:, iv, 0] = model.params[init_vars_[iv]][:,0]
            else:
                IC_[:, iv, :] = delay_state_vars_[:, iv, :]
        else:
            IC_[:, iv, 0] = model.params[init_vars_[iv]][:]

    return IC_

def set_init(model, IC_, init_vars_, state_vars_, startind_):
    for iv in range(len(init_vars_)):
        if model.params[init_vars_[iv]].ndim == 1:
            model.params[init_vars_[iv]][:] = IC_[:, iv, 0]
        else:
            if startind_ == 1:
                model.params[init_vars_[iv]][:,0] = IC_[:, iv, 0] 
            else:
                model.params[init_vars_[iv]] = IC_[:, iv, :]

def set_init_1d(IC_, model):
    state_vars = model.state_vars
    init_vars = model.init_vars
    for iv in range(len(init_vars)):
        for sv in range(len(state_vars)):
            if state_vars[sv] in init_vars[iv]:
                if model.params[init_vars[iv]].ndim == 2:
                    model.params[init_vars[iv]][0,:] = IC_[sv]
                else:
                    model.params[init_vars[iv]][0] = IC_[sv]

def update_init(model, init_vars_, state_vars_):
    for iv, sv in zip( range(len(init_vars_)), range(len(state_vars_)) ):
        if state_vars_[sv] in init_vars_[iv]:
            if ( type(model.params[init_vars_[iv]]) == np.float64 or type(model.params[init_vars_[iv]]) == float ):
                model.params[init_vars_[iv]] = model.state[state_vars_[sv]][:,-1]
            elif model.params[init_vars_[iv]].ndim == 2:
                model.params[init_vars_[iv]][:,0] = model.state[state_vars_[sv]][:,-1]
            else:
                model.params[init_vars_[iv]] = model.state[state_vars_[sv]][:,-1]
                    
def update_init_delayed(model, delay_state_vars_, init_vars_, state_vars_, t_pre_ndt_, startind_):
    for iv, sv in zip( range(len(init_vars_)), range(len(state_vars_)) ):
        if state_vars_[sv] in init_vars_[iv]:
            if model.params[init_vars_[iv]].ndim == 2:
                if (t_pre_ndt_ < startind_):
                    delay_state_vars_[:, sv, :-1] = model.params[init_vars_[iv]][:,1:]
                    delay_state_vars_[:, sv, -t_pre_ndt_:] = model.state[state_vars_[sv]][:,:]
                    model.params[init_vars_[iv]][:,:] = delay_state_vars_[:, sv, :]
                else:
                    model.params[init_vars_[iv]] = model.state[state_vars_[sv]][:,-startind_:]
                    delay_state_vars_[:, sv, :] = model.state[state_vars_[sv]][:,-startind_:]
            else:
                model.params[init_vars_[iv]] = model.state[state_vars_[sv]]
   
@numba.njit
def setmaxcontrol(n_control_vars, control_, max_control_, min_control_):
    for j in range(len(control_[0,0,:])):
        for v in range(n_control_vars):
            #print("set control max ", v, max_control_[v], min_control_[v])
            if control_[0,v,j] > max_control_[v]:
                control_[0,v,j] = max_control_[v]
            elif control_[0,v,j] < min_control_[v]:
                control_[0,v,j] = min_control_[v]
    return control_

@numba.njit
def scalemaxcontrol(control_, max_control_, min_control_):
    max_val_ = np.amax(control_)
    if max_val_ != 0 and max_val_> max_control_:
        scale_factor_ = np.abs(max_control_ / max_val_)
        control_ *= scale_factor_
    min_val_ = np.amin(control_)
    if min_val_ != 0. and min_control_ != 0. and min_val_< min_control_:
        scale_factor_ = np.abs(min_control_ / min_val_)
        control_ *= scale_factor_
    return control_

def StrongWolfePowellLineSearch():
    # would require to compute new gradient for each step
    return None

# backtracking according to Armijo-Goldstein
def AG_line_search(model, N, V, T, dt, state_, target_, control_, dir_, ip_, ie_, is_, start_step_ = 20., max_it_ = 1000,
              bisec_factor_ = 0.5, max_control_ = [20., 20., 0.2, 0.2], min_control_ = [-20., -20., 0., 0.],
              tolerance_ = 1e-16, substep_ = 0.1, variables_ = [0,1], alg = "A1", control_parameter = 0.02, grad_ = None):
    
    m_ = np.zeros(( N, V ))
    for n_ in range(N):
        for v_ in range(V):
            m_[n_, v_] = np.dot(dir_[n_,v_,:], grad_[n_,v_,:])
    
    t_ = - control_parameter * np.sum(m_)
    cost0_ = cost.f_int(N, V, T, dt, state_, target_, control_, ip_, ie_, is_, v_ = variables_)
    step_ = start_step_
    
    for j in range(max_it_):
                
        test_control_ = control_ + step_ * dir_
        test_control_ = setmaxcontrol(V, test_control_, max_control_, min_control_)
        state1_ = updateState(model, test_control_)
        cost1_ = cost.f_int(N, V, T, dt, state1_, target_, test_control_, v_ = variables_)
        
        #print("cost0 = ", cost0_)
        #print("cost1 = ", cost1_)
        #print("step, t_ = ", step_, t_)
        
        if cost0_ - cost1_ - step_ * t_ >= 0.:
            return step_, cost1_, start_step_
        
        step_ *= bisec_factor_
        
    print("max iteration reached, condition not satisfied")
    return 0., cost0_, start_step_
    
    
    
def step_size(model, N, V, T, dt, state_, target_, control_, dir_, ip_, ie_, is_, start_step_ = 20., max_it_ = 1000,
              bisec_factor_ = 2., max_control_ = [20., 20., 0.2, 0.2], min_control_ = [-20., -20., 0., 0.],
              tolerance_ = 1e-32, substep_ = 0.1, variables_ = [0,1], alg = "A1", control_parameter = 0.2, grad_ = None, noise_real=0, init_vars=None, startind_=0):

    step_lim = 1e-20
    step_lim_up = 1e10
    
    if noise_real == 0:
        cost0_int_ = cost.f_int(N, V, T, dt, state_, target_, control_, ip_, ie_, is_, v_ = variables_)
    else:
        cost0_int_ = 0.
        for noise_ in range(noise_real):
            cost0_int_ += cost.f_int(N, V, T, dt, state_[noise_][:,:2,:], target_, control_, ip_, ie_, is_, v_ = variables_)
        cost0_int_ /= noise_real

    
    #print("first cost = ", cost0_int_)
    cost_min_int_ = cost0_int_
    step_ = start_step_
    step_min_ = 0.
    
    factor = 2.**7
        
    start_step_out_ = start_step_
    
    #print("try again")
    
    cost_list = []

    for i in range(max_it_):
        test_control_ = control_ + step_ * dir_
        # include maximum control value to assure no divergence
        test_control_ = setmaxcontrol(V, test_control_, max_control_, min_control_)

        if noise_real == 0:
            if type(init_vars) != type(None):
                #print(init_vars)
                set_init(model, init_vars, model.init_vars, model.state_vars, startind_)
            state1_ = updateState(model, test_control_)
            cost1_int_ = cost.f_int(N, V, T, dt, state1_, target_, test_control_, ip_, ie_, is_, v_ = variables_)
        else:
            cost1_int_ = 0.
            cost_list = []
            for noise_ in range(noise_real):
                set_init(model, init_vars[noise_], model.init_vars, model.state_vars, startind_)
                state1_ = updateState(model, test_control_)
                cost1_int_ += cost.f_int(N, V, T, dt, state1_, target_, control_, ip_, ie_, is_, v_ = variables_)
                cost_list.append(cost1_int_)
            cost1_int_ /= noise_real
        
        #print("test control = ", test_control_[0,0,:])
        #print("step, cost, initial cost = ", step_, cost1_int_, cost0_int_)
        
        if step_ * np.amax(np.absolute(dir_)) < tolerance_:
            #print("test control change smaller than tolerance, return zero step")
            return 0., cost0_int_, start_step_

        if (cost1_int_ < cost_min_int_):
            cost_min_int_ = cost1_int_
            step_min_ = step_
            
        # return smallest step size before cost is increasing again
        elif (cost1_int_ > cost_min_int_ and cost_min_int_ < cost0_int_):
            
            if (i == 1 and alg == "A1"):
                step_ = factor * start_step_
                #print("too small start step, increase to ", step_)
                return step_size(model, N, V, T, dt, state_, target_, control_, dir_, ip_, ie_, is_,
                                 start_step_ = step_, max_it_ = max_it_,
                                 bisec_factor_ = bisec_factor_, max_control_ = max_control_, min_control_ = min_control_,
                                 tolerance_ = tolerance_, substep_ = substep_, variables_ = variables_, noise_real=noise_real, init_vars=init_vars, startind_=startind_)
            elif (step_ < start_step_ / (2. * factor) and alg == "A1"):
                start_step_ /= factor
                #print("too large start step, decrease to ", start_step_)
            

            # iterate between step_range[0] and [2] more granularly
            #if noise_real > 1:
            #    print("std dev cost, step ", np.std(cost_list), step_min_, np.std(cost_list) * step_min_)

            substep = substep_
            
            step_min_up, cost_min_int_, cost_list_up = scan(model, N, V, T, dt, substep, control_, step_min_, dir_, target_,
                                                  cost_min_int_, max_control_, min_control_, ip_, ie_, is_, variables_, noise_real=noise_real, init_vars=init_vars, startind_=startind_)

            substep = - substep_
            step_min_down, cost_min_int_, cost_list_down = scan(model, N, V, T, dt, substep, control_, step_min_, dir_, target_,
                                                cost_min_int_, max_control_, min_control_, ip_, ie_, is_, variables_, noise_real=noise_real, init_vars=init_vars, startind_=startind_)

            result = [0., 0., 0.]
            if (step_min_up > step_min_ ):
                if (step_min_down == step_min_):
                    result = step_min_up, cost_min_int_, start_step_
                    result_n = step_min_up, cost_list_up, start_step_
                elif (step_min_down < step_min_):
                    result = step_min_down, cost_min_int_, start_step_
                    result_n = step_min_down, cost_list_down, start_step_
            elif (step_min_down < step_min_):
                result = step_min_down, cost_min_int_, start_step_
                result_n = step_min_down, cost_list_down, start_step_
            else:
                result = [step_min_, cost_min_int_, start_step_]
                result_n = [step_min_, cost_list, start_step_]
                            
            if noise_real > 0:
                return result_n
            else:
                return result
        
        if (i == max_it_-1):
            if (max_it_ != 1):
                print(" max iteration reached, step size = ", step_)
            return step_min_, cost_min_int_, start_step_
        
        if step_ > step_lim and step_ < step_lim_up:
            step_ /= bisec_factor_
        else:
            print("step size too small or too large")
            return 0., cost0_int_, start_step_
        
def scan(model_, N, V, T, dt_, substep_, control_, step_min_, dir_, target_, cost_min_int_, max_control_,
         min_control_, ip_, ie_, is_, variables_ = [0,1], noise_real=0, init_vars=None, startind_=0):

    i = 1.
    cntrl_ = control_ + ( 1. + i * substep_ ) * step_min_ * dir_
    cntrl_ = setmaxcontrol(V, cntrl_, max_control_, min_control_)
    
    if noise_real == 0:
        if type(init_vars) != type(None):
            set_init(model_, init_vars, model_.init_vars, model_.state_vars, startind_)
        state_ = updateState(model_, cntrl_)
        cost_int = cost.f_int(N, V, T, dt_, state_, target_, cntrl_, ip_, ie_, is_, v_ = variables_)
    else:
        cost_int = 0.
        for noise_ in range(noise_real):
            set_init(model_, init_vars[noise_], model_.init_vars, model_.state_vars, startind_)
            state_ = updateState(model_, cntrl_)
            cost_int += cost.f_int(N, V, T, dt_, state_, target_, cntrl_, ip_, ie_, is_, v_ = variables_)
        cost_int /= noise_real
    
    step_min1_ = step_min_

    cost_list_scan = []
    
    while (i <= 10. and cost_int < cost_min_int_):
        i += 1.
        cost_min_int_ = cost_int
        step_min1_ += substep_ * step_min_
        
        # new control
        cntrl_ = control_ + ( 1. + i * substep_ ) * step_min_ * dir_
        cntrl_ = setmaxcontrol(V, cntrl_, max_control_, min_control_)

        cost_list_scan = []
        if noise_real == 0:
            if type(init_vars) != type(None):
                set_init(model_, init_vars, model_.init_vars, model_.state_vars, startind_)
            state_ = updateState(model_, cntrl_)
            cost_int = cost.f_int(N, V, T, dt_, state_, target_, cntrl_, ip_, ie_, is_, v_ = variables_)
            cost_list_scan.append(cost_int)
        else:
            
            for noise_ in range(noise_real):
                set_init(model_, init_vars[noise_], model_.init_vars, model_.state_vars, startind_)
                state_ = updateState(model_, cntrl_)
                cost_int += cost.f_int(N, V, T, dt_, state_, target_, cntrl_, ip_, ie_, is_, v_ = variables_)
                cost_list_scan.append(cost_int)
                
            cost_int /= noise_real    
        
    return step_min1_, cost_min_int_, cost_list_scan


def step_noise_1(line_search_func, model, noise_real, N, n_control_vars, T, dt, state0_, target_state_,
                            best_control_, d_, ip_, ie_, is_, cost_init, startstep_,
                            cntrl_max_, cntrl_min_, prec_vars, grad1_, init_vars_, startind_):

    repeat = True
    limit_multiple0 = 3.

    while repeat:
    
        s_ = np.zeros(( noise_real ))
        tc_ = [-1] * noise_real
        ss_ = [startstep_] * noise_real
        n_ = 0

        limit_multiple = 3.

        while n_ < noise_real:
            
            #print("Noise step ", n_)
                
            s_[n_], tc_[n_], ss_[n_] = line_search_func(model, N, n_control_vars, T, dt, state0_[n_][:,:2,:], target_state_,
                                best_control_, d_, ip_, ie_, is_, start_step_ = ss_[n_], max_it_ = 500,
                                max_control_ = cntrl_max_, min_control_ = cntrl_min_, variables_ = prec_vars, grad_ = grad1_, noise_real=0, init_vars=init_vars_[n_], startind_=startind_)

            #print(cost_init, tc_[n_])
            if tc_[n_] <= limit_multiple * np.mean(cost_init):
                n_ += 1
            else:
                limit_multiple *= 1.5
                

        s_mean = 0.
        ss_mean = 0.
        n_s = 0.
        for noise_ in range(noise_real):
            if s_[noise_] != 0.:
                s_mean += s_[noise_]
                ss_mean += ss_[noise_]
                n_s += 1
        if n_s > 0:
            s_mean /= n_s
            ss_mean /= n_s

        cntrl_ = best_control_ + s_mean * d_
        cntrl_ = setmaxcontrol(cntrl_.shape[1], cntrl_, cntrl_max_, cntrl_min_)

        """
        max_jump = .5
        for v_ in range(cntrl_.shape[1]):
            for t_ in range(1,cntrl_.shape[2]):
                cntrl_diff = np.abs(cntrl_[0,n_,t_] - cntrl_[0,n_,t_-1])
                if cntrl_diff > max_jump:
                    #print("jump, repeat")
                    continue
        """

        for noise_ in range(noise_real):
            set_init(model, init_vars_[noise_], model.init_vars, model.state_vars, startind_)
            state_ = updateState(model, cntrl_)
            #plt.plot(model.t, model.rates_exc[0,:], color='red')
            #plt.plot(model.t, model.rates_inh[0,:], color='blue')
            #plt.show()
            tc_[noise_] = cost.f_int(N, n_control_vars, T, dt, state_, target_state_, cntrl_, ip_, ie_, is_, v_ = prec_vars)


        if np.mean(tc_) <= limit_multiple0 * np.mean(cost_init):
            repeat = False
        else:
            #print("repeat")
            limit_multiple0 += 1.

    return s_mean, tc_, ss_mean

def step_noise_2(line_search_func, model, noise_real, N, n_control_vars, T, dt, state0_, target_state_,
                            best_control_, d_, ip_, ie_, is_, cost_init, startstep_,
                            cntrl_max_, cntrl_min_, prec_vars, grad1_, init_vars_, startind_):

    s_, tc_, ss_ = line_search_func(model, N, n_control_vars, T, dt, state0_, target_state_,
                            best_control_, d_, ip_, ie_, is_, start_step_ = startstep_, max_it_ = 500,
                            max_control_ = cntrl_max_, min_control_ = cntrl_min_, variables_ = prec_vars, grad_ = grad1_, noise_real=noise_real, init_vars=init_vars_, startind_=startind_)

    #print(s_)
    #print(tc_)
    #print(ss_)

    return s_, tc_, ss_

def get_step_noise(line_search_func, model, N, n_control_vars, T, dt, noise_real, state0_, target_state_,
                    best_control_, dir0_, ip_, ie_, is_, startstep_exc_, startstep_inh_, start_step_noise_, cntrl_max_,
                    cntrl_min_, grad1_, cost_init, init_vars, startind, prec_vars=[0], control_vars=[0,1], separate_comp=True, noise_in_step_comp = False):
            
    if separate_comp:
        # compute stepsize separately and then put together

        tc_exc = -1
        tc_inh = -1

        if 0 in control_vars and len(control_vars) > 1:
            d_exc = dir0_.copy()
            d_exc[:,1:,:] = 0.

            if noise_in_step_comp == False:
                s_exc, tc_exc, startstep_exc_ = step_noise_1(line_search_func, model, noise_real, N, n_control_vars, T, dt, state0_, target_state_,
                            best_control_, d_exc, ip_, ie_, is_, cost_init, startstep_exc_,
                            cntrl_max_, cntrl_min_, prec_vars, grad1_, init_vars, startind)

            else:
                s_exc, tc_exc, startstep_exc_ = step_noise_2(line_search_func, model, noise_real, N, n_control_vars, T, dt, state0_, target_state_,
                            best_control_, d_exc, ip_, ie_, is_, cost_init, startstep_exc_,
                            cntrl_max_, cntrl_min_, prec_vars, grad1_, init_vars, startind)
        
        if 1 in control_vars and len(control_vars) > 1:
            d_inh = dir0_.copy()
            d_inh[:,0,:] = 0.

            if noise_in_step_comp == False:
                s_inh, tc_inh, startstep_inh_ = step_noise_1(line_search_func, model, noise_real, N, n_control_vars, T, dt, state0_, target_state_,
                            best_control_, d_inh, ip_, ie_, is_, cost_init, startstep_inh_,
                            cntrl_max_, cntrl_min_, prec_vars, grad1_, init_vars, startind)

            else:
                s_inh, tc_inh, startstep_inh_ = step_noise_2(line_search_func, model, noise_real, N, n_control_vars, T, dt, state0_, target_state_,
                            best_control_, d_exc, ip_, ie_, is_, cost_init, startstep_exc_,
                            cntrl_max_, cntrl_min_, prec_vars, grad1_, init_vars, startind)


        #if 0 in control_variables and 1 in control_variables:
        #    joint_dir = dir0_.copy()
        #    joint_dir[:,0,:] = s_exc * dir0_[:,0,:] #/ (s_exc + s_inh)
        #    joint_dir[:,1,:] = s_inh * dir0_[:,1,:] #/ (s_exc + s_inh)
            
        #    joint_step_, joint_cost, startstep_joint_ = line_search_func(model, N, n_control_vars, T, dt, state1_[:,:2,:],
        #                    target_state_, best_control_, joint_dir, ip_, ie_, is_, start_step_ = startstep_joint_, max_it_ = 500,
        #                    max_control_ = cntrl_max_, min_control_ = cntrl_min_, variables_ = prec_variables, grad_ = grad1_)
        #    minCost.append(joint_cost)


    if noise_in_step_comp == False:
        s_noise, tc_noise, start_step_noise_ = step_noise_1(line_search_func, model, noise_real, N, n_control_vars, T, dt, state0_, target_state_,
                    best_control_, dir0_, ip_, ie_, is_, cost_init, start_step_noise_,
                    cntrl_max_, cntrl_min_, prec_vars, grad1_, init_vars, startind)
    
    else:
        s_noise, tc_noise, start_step_noise_ = step_noise_2(line_search_func, model, noise_real, N, n_control_vars, T, dt, state0_, target_state_,
                    best_control_, dir0_, ip_, ie_, is_, cost_init, start_step_noise_,
                    cntrl_max_, cntrl_min_, prec_vars, grad1_, init_vars, startind)
                
    if separate_comp:
        mean_cost_exc = np.mean(tc_exc) 
        mean_cost_inh = np.mean(tc_inh) 
        mean_cost_ = np.mean(tc_noise)
        mean_cost_list = []
        for c_ in [ mean_cost_exc, mean_cost_inh, mean_cost_ ]:
            if c_ != -1:
                mean_cost_list.append(c_)
        cost_min = min( mean_cost_list )

        #print("cost exc ", mean_cost_exc)
        #print("cost inh ", mean_cost_inh)
        #print("cost, step normal ", mean_cost_, s_noise)
        #print("previous cost ", np.mean(cost_init) )

        #if cost_min >= np.mean(cost_init):
        #    print("no improvement")
        #    step_ = 0.
        #    cost_ = cost_init

        #else:
        #print(tc_exc)
        #print(tc_inh)
        #print(tc_noise)
        #print(cost_min, mean_cost_exc, mean_cost_inh, mean_cost_)

        if cost_min ==  mean_cost_exc:
            #print("choose exc only")
            step_ = s_exc
            cost_ = tc_exc
            dir0_ = d_exc.copy()
            
        elif cost_min ==  mean_cost_inh:
            #print("choose inh only")
            step_ = np.mean(s_inh)
            cost_ = tc_inh
            dir0_ = d_inh.copy()

        elif cost_min ==  mean_cost_:
            #print("choose normal")
            step_ = np.mean(s_noise)
            cost_ = tc_noise
        
        
        #elif (joint_cost ==  impMax):
        #    print("choose exc, inh combination")
        #    step_ = joint_step_
        #    total_cost_[i] = joint_cost
        #    dir0_ = joint_dir.copy()
        #    startStep_ = startstep_joint_

    else:
        #print("choose noraml")
        step_ = s_noise

    #print(step_)

    return step_, dir0_, cost_

def set_pre_post(i1, i2, bc_, bs_, best_control_, state_pre_, state_, state_post_, state_vars, a, b):
    
    if (i2 != 0 and i1 != 0):   
        bc_[:,:,i1:-i2] = best_control_[:,:,:]
        bs_[:,:,:i1+1] = state_pre_[:,:,:]
        check_pre(i1, bs_, state_, state_vars, a, b)
        bs_[:,:,i1:-i2] = state_[:,:,:]
        if i2 == 1:
            bs_[:,:,-i2] = state_post_[:,:,1]
        else:
            bs_[:,:,-i2:] = state_post_[:,:,1:]
        check_post(i2, bs_, state_post_, state_vars, a, b)
        
    elif (i2 == 0 and i1 != 0):
        bc_[:,:,i1:] = best_control_[:,:,:]
        bs_[:,:,:i1+1] = state_pre_[:,:,:]
        check_pre(i1, bs_, state_, state_vars, a, b)
        bs_[:,:,i1:] = state_[:,:,:]
        
    elif (i2 != 0 and i1 == 0):
        bc_[:,:,:-i2] = best_control_[:,:,:]
        bs_[:,:,:-i2] = state_[:,:,:]
        bs_[:,:,-i2:] = state_post_[:,:,:]
        check_post(i2, bs_, state_post_, state_vars, a, b)
                    
    else:
        bc_[:,:,:] = best_control_[:,:,:]
        bs_[:,:,:] = state_[:,:,:]
        
    return bc_, bs_

def check_pre(i1, bs_, state_, state_vars, a, b):
    for n in range(bs_.shape[0]):
        for v in range(bs_.shape[1]):
            if state_vars[v] == "Vmean_exc":
                if np.abs(bs_[n,v,i1] - state_[n,v,0]) > 1.:
                    logging.error("Problem in initial value trasfer pre")
                    print("Problem in initial value trasfer pre: ", state_vars[v], bs_[n,v,i1], state_[n,v,0])
            elif np.abs(bs_[n,v,i1] - state_[n,v,0]) > 1e-8:
                logging.error("Problem in initial value trasfer pre")
                print("Problem in initial value trasfer pre: ", state_vars[v], bs_[n,v,i1], state_[n,v,0])
                
def check_post(i2, bs_, state_post_, state_vars, a, b):
    for n in range(bs_.shape[0]):
        for v in range(bs_.shape[1]):
            if state_vars[v] == "Vmean_exc":
                if np.abs(bs_[n,v,-i2-1] - state_post_[n,v,0]) > 1.:
                    logging.error("Problem in initial value trasfer post")
                    print("Problem in initial value trasfer post: ", state_vars[v], bs_[n,v,-i2-1], state_post_[n,v,0])
            elif np.abs(bs_[n,v,-i2-1] - state_post_[n,v,0]) > 1e-8:
                logging.error("Problem in initial value trasfer post")
                print("Problem in initial value trasfer post: ", state_vars[v], bs_[n,v,-i2-1], state_post_[n,v,0])
                

def adapt_step(control_, ind_node, ind_var, start_step_, dir_, max_control_, min_control_):
    start_st_ = start_step_
    max_index = -1
    min_index = -1
    max_cntrl = max_control_
    min_cntrl = min_control_
    
    for k in range(control_.shape[2]):
        if ( control_[ind_node,ind_var,k] + start_step_ * dir_[ind_node,ind_var,k] > max_cntrl ):
            max_index = k
            max_cntrl = control_[ind_node, ind_var,k] + start_step_ * dir_[ind_node, ind_var,k]
        elif ( control_[ind_node,ind_var,k] + start_step_ * dir_[ind_node,ind_var,k] < min_cntrl ):
            min_index = k
            min_cntrl = control_[ind_node, ind_var,k] + start_step_ * dir_[ind_node, ind_var,k]
    if max_index != -1:
        start_st_ = ( max_control_ - control_[ind_node,ind_var,max_index] ) / dir_[ind_node,ind_var,max_index]
    elif min_index != -1:
        start_st_ = ( min_control_ - control_[ind_node,ind_var,min_index] ) / dir_[ind_node,ind_var,min_index]
    return start_st_

# update rule for direction according to Hestenes-Stiefel
def betaHS(N, n_control_vars, grad0_, grad1_, dir0_):
    betaHS = np.zeros(( N, n_control_vars ))
    for n in range(N):
        for v in range(n_control_vars):
            numerator = np.dot( grad1_[n,v,:], ( grad1_[n,v,:] - grad0_[n,v,:] ) )
            denominator = np.dot( dir0_[n,v,:], ( grad1_[n,v,:] - grad0_[n,v,:] ) )
            #print("numerator = ", numerator)
            #print("denominator = ", denominator)
            if np.abs(denominator) > 1e-6 :
                betaHS[n,v] = numerator / denominator
    return betaHS

# update rule for direction according to Fletcher-Reeves
def betaFR(N, n_control_vars, grad0_, grad1_):
    betaFR = np.zeros(( N, n_control_vars ))
    for n in range(N):
        for v in range(n_control_vars):
            numerator = np.dot( grad1_[n,v,:], grad1_[n,v,:] )
            denominator = np.dot( grad0_[n,v,:], grad0_[n,v,:] )
            if np.abs(denominator) > 1e-6 :
                betaFR[n,v] = numerator / denominator
    return betaFR

# update rule for direction according to Polak-Ribiere
def betaPR(N, n_control_vars, grad0_, grad1_):
    betaPR = np.zeros(( N, n_control_vars ))
    for n in range(N):
        for v in range(n_control_vars):
            numerator = np.dot( grad1_[n,v,:], ( grad1_[n,v,:] - grad0_[n,v,:] ) )
            denominator = np.dot( grad0_[n,v,:], grad0_[n,v,:] )
            if np.abs(denominator) > 1e-6 :
                betaPR[n,v] = numerator / denominator
    return betaPR

# update rule for direction "conjugate descent"
def betaCD(N, n_control_vars, grad0_, grad1_, dir0_):
    betaCD = np.zeros(( N, n_control_vars ))
    for n in range(N):
        for v in range(n_control_vars):
            numerator = np.dot( grad1_[n,v,:], grad1_[n,v,:] )
            denominator = np.dot( dir0_[n,v,:], grad0_[n,v,:] )
            if np.abs(denominator) > 1e-6 :
                betaCD[n,v] = - numerator / denominator
    return betaCD

# update rule for direction according to Liu-Storey
def betaLS(N, n_control_vars, grad0_, grad1_, dir0_):
    betaLS = np.zeros(( N, n_control_vars ))
    for n in range(N):
        for v in range(n_control_vars):
            numerator = np.dot( grad1_[n,v,:], ( grad1_[n,v,:] - grad0_[n,v,:] ) )
            denominator = np.dot( dir0_[n,v,:], grad0_[n,v,:] )
            if np.abs(denominator) > 1e-6 :
                betaLS[n,v] = - numerator / denominator
    return betaLS

# update rule for direction according to Dai-Yuan
def betaDY(N, n_control_vars, grad0_, grad1_, dir0_):
    betaDY = np.zeros(( N, n_control_vars ))
    for n in range(N):
        for v in range(n_control_vars):
            numerator = np.dot( grad1_[n,v,:], grad1_[n,v,:] )
            denominator = np.dot( dir0_[n,v,:], ( grad1_[n,v,:] - grad0_[n,v,:] ) )
            if np.abs(denominator) > 1e-6 :
                betaDY[n,v] = numerator / denominator
    return betaDY

# update rule for direction according to Wei et al.
def betaWYL(N, n_control_vars, grad0_, grad1_):
    betaWYL = np.zeros(( N, n_control_vars ))
    for n in range(N):
        for v in range(n_control_vars):
            g0abs = np.sqrt( np.dot( grad0_[n,v,:], grad0_[n,v,:] ) )
            g1abs = np.sqrt( np.dot( grad1_[n,v,:], grad1_[n,v,:] ) )
            numerator = np.dot( grad1_[n,v,:], grad1_[n,v,:] - grad0_[n,v,:] * ( g1abs / g0abs ) )
            denominator = np.dot( grad0_[n,v,:], grad0_[n,v,:] )
            if np.abs(denominator) > 1e-6 :
                betaWYL[n,v] = numerator / denominator
    return betaWYL

# update rule for direction according to Hager-Zhang
def betaHZ(N, n_control_vars, grad0_, grad1_, dir0_):
    betaHZ = np.zeros(( N, n_control_vars ))
    eta = 0.01
    for n in range(N):
        for v in range(n_control_vars):
            diff = grad1_[n,v,:] - grad0_[n,v,:]
            denominator = np.dot( dir0_[n,v,:], diff )
            if np.abs(denominator) > 1e-6 :
                beta0 = np.dot( diff - 2. * np.dot( diff, diff ) * dir0_[n,v,:] / denominator,
                           grad1_[n,v,:] / denominator )
            else:
                beta0 = - 1e10
            numerator = np.dot( grad0_[n,v,:], grad0_[n,v,:] )
            denominator = np.dot( dir0_[n,v,:], dir0_[n,v,:] )
            if np.abs(denominator) > 1e-6 :
                eta0 = - min( eta, np.sqrt( numerator ) ) / np.sqrt( denominator )
            else:
                eta0 = 0.
            betaHZ[n,v] = max(beta0, eta0)
    return betaHZ

def compute_gradient(N, n_control_vars, T, dt, best_control_, grad1_, phi1_, control_variables, ie_, is_):
    grad_cost_e_ = cost.cost_energy_gradient(best_control_, ie_)
    grad_cost_s_ = cost.cost_sparsity_gradient(N, n_control_vars, T, dt, best_control_, is_)
        
    for n in range(N):
        if N == 1:
            c_var = control_variables
        else:
            c_var = control_variables[n]
        for j in range(n_control_vars):
            if j in c_var:
                #print("j, adjoint, energy, sparsity gradient = ", j)
                #print(phi1_[:,j,:20])
                #print(grad_cost_e_[:,j,:20])
                #print(grad_cost_s_[:,j,:20])
            
                grad1_[n,j,:] = grad_cost_e_[n,j,:] + grad_cost_s_[n,j,:] + phi1_[n,j,:]
                #grad1_[n,j,:] = phi1_[n,j,:]
    return grad1_

def set_direction(N, T, n_control_vars, grad0_, grad1_, dir0_, i, CGVar, tolerance_):
        
    beta = np.zeros(( N, n_control_vars ))
    
    if (i >= 2 and CGVar != None):
        if CGVar == "HS":        # Hestens-Stiefel
            beta = betaHS(N, n_control_vars, grad0_, grad1_, dir0_)
        elif CGVar == "FR":        # Fletcher-Reeves
            beta = betaFR(N, n_control_vars, grad0_, grad1_)
        elif CGVar == "PR":        # Polak-Ribiere
            beta = betaPR(N, n_control_vars, grad0_, grad1_)
        elif CGVar == "CD":        # conjugate descent
            beta = betaCD(N, n_control_vars, grad0_, grad1_, dir0_)
        elif CGVar == "LS":        # Liu-Storey
            beta = betaLS(N, n_control_vars, grad0_, grad1_, dir0_)
        elif CGVar == "DY":        # Dai-Yuan
            beta = betaDY(N, n_control_vars, grad0_, grad1_, dir0_)
        elif CGVar == "WYL":        # Wei et al.
            beta = betaWYL(N, n_control_vars, grad0_, grad1_)
        elif CGVar == "HZ":        # Hager-Zhang
            beta = betaHZ(N, n_control_vars, grad0_, grad1_, dir0_)
            
    dir1_ = np.zeros(( N, n_control_vars, T ))
    for n in range(N):
        for v in range(n_control_vars):
            dir1_[n,v,:] = beta[n,v] * dir0_[n,v,:]
    
    dir0_ = - grad1_.copy() + dir1_
    
    # if this is too close to zero, use beta = 0 instead
    if (CGVar != None and np.amax(np.absolute(dir0_)) < tolerance_ ):
        print("Descent direction vanishing, use standard gradient descent")
        dir0_ = - grad1_.copy()
        
    return dir0_
    
    
"""  
def test_step(model, N, V, T, dt, state_, target_, control_, dir_, cost0_, ip_, ie_, is_, test_step_ = 1e-12, prec_variables_ = [0,1]):
    cost0_int_ = cost0_
    
    test_control_ = control_ + test_step_ * dir_
    state1_ = updateState(model, test_control_)
    cost1_int_ = cost.f_int(N, V, T, dt, state1_, target_, test_control_, ip_, ie_, is_, v_ = prec_variables_)
    #print("test step size computation : ------ step size, cost1, cost0 : ", test_step_, cost1_int_, cost0_int_)
        
    if (cost1_int_ < cost0_int_):
        return test_step_, cost1_int_
    else:
        return 0., cost0_int_
"""