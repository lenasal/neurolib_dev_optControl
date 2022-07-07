import numpy as np
import logging
import numba
from numba.typed import List
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from . import costFunctions as cost
from . import func_optimize as fo
from ..models import jacobian_aln as jac_aln

np.set_printoptions(precision=8)

model_name = "-aln"


VALID_VAR = {None, "HS", "FR", "PR", "CD", "LS", "DY", "WYL", "HZ"}
VALID_LS = {None, "AG"}

def M4(model, noise_real, init_params, control_, target_state, c_scheme_, u_mat_, u_scheme_, max_iteration_, tolerance_, startStep_,
       cntrl_max_, cntrl_min_, t_sim_, t_sim_pre_, t_sim_post_, method_step,
       CGVar = None, line_search_ = None, control_variables_ = [0,1], prec_variables_ = [0,1], separate_comp = True, transition_time_ = 0.):

    if method_step == 'S1':
        noise_in_step_comp_=False
    elif method_step == 'S2':
        noise_in_step_comp_=True
        
    dt = model.params['dt']
    max_iteration_ = int(max_iteration_)
        
    prec_variables = List()
    for v in prec_variables_:
        prec_variables.append(v)
                
    control_variables = List()
    for v in control_variables_:
        control_variables.append(v)
        
    n_maxDelay = model.getMaxDelay()
        
    startind_ = int(n_maxDelay + 1)
    target_state_ = target_state.copy()
            
    state_vars = model.state_vars
    init_vars = model.init_vars
    n_control_vars = len(model.control_input_vars)
        
    T = int( 1 + np.around(t_sim_ / dt, 1) )
    
    V = len(state_vars)
    V_target = target_state_.shape[1]
    i=0
       
    ##############################################
    # PARAMETERS FOR JACOBIAN
    # TODO: time dependent exc current
    ext_exc_current = model.params.ext_exc_current
    ext_inh_current = model.params.ext_inh_current
    
    # ee, ei, ie, ii
    ext_ee_rate = model.params.ext_ee_rate
    ext_ei_rate = model.params.ext_ei_rate
    ext_ie_rate = model.params.ext_ie_rate
    ext_ii_rate = model.params.ext_ii_rate
    
    sigmae_ext = model.params.sigmae_ext
    sigmai_ext = model.params.sigmai_ext
    
    a = model.params["a"]
    b = model.params["b"]
    tauA = model.params["tauA"]
    
    C = model.params["C"]
    c_gl = model.params["c_gl"]
    Ke_gl = model.params["Ke_gl"]
    Ki_gl = model.params["Ki_gl"]
    
    Ke = model.params["Ke"]
    Ki = model.params["Ki"]
    tau_se = model.params["tau_se"] 
    tau_si = model.params["tau_si"] 
    cee = model.params["cee"]
    cei = model.params["cei"]
    cie = model.params["cie"]
    cii = model.params["cii"]
    Jee_max = model.params["Jee_max"]
    Jei_max = model.params["Jei_max"]
    Jie_max = model.params["Jie_max"]
    Jii_max = model.params["Jii_max"]
    taum = model.params.C / model.params.gL
    
    tau_se_sq = tau_se**2
    tau_si_sq = tau_si**2
    Jee_sq = Jee_max**2
    Jei_sq = Jei_max**2
    Jie_sq = Jie_max**2
    Jii_sq = Jii_max**2
    
    tau_ou = model.params.tau_ou
    
    N = model.params.N
    
    ndt_de = np.around(model.params.de / dt).astype(int)
    ndt_di = np.around(model.params.di / dt).astype(int)
    
    factor_ee1 = ( cee * Ke * tau_se / np.abs(Jee_max) )
    factor_ee2 = ( cee**2 * Ke * tau_se_sq / Jee_sq )

    factor_ei1 = ( cei * Ki * tau_si / np.abs(Jei_max) )
    factor_ei2 = ( cei**2 * Ki * tau_si_sq / Jei_sq )
    
    factor_ie1 = ( cie * Ke * tau_se / np.abs(Jie_max) )
    factor_ie2 = ( cie**2 * Ke * tau_se_sq / Jie_sq )
   
    factor_ii1 = ( cii * Ki * tau_si / np.abs(Jii_max) )
    factor_ii2 = ( cii**2 * Ki * tau_si_sq / Jii_sq )
    
    factor_eec1 = c_gl * Ke_gl * tau_se / np.abs(Jee_max)
    factor_eec2 = c_gl**2 * Ke_gl * tau_se_sq / Jee_sq 
    
    factor_eic1 = c_gl * Ki_gl * tau_si / np.abs(Jei_max)
    factor_eic2 = c_gl**2 * Ki_gl * tau_si_sq / Jei_sq 
    
    factor_iec1 = c_gl * Ke_gl * tau_se / np.abs(Jie_max)
    factor_iec2 = c_gl**2 * Ke_gl * tau_se_sq / Jie_sq 
    
    factor_iic1 = c_gl * Ki_gl * tau_si / np.abs(Jii_max)
    factor_iic2 = c_gl**2 * Ki_gl * tau_si_sq / Jii_sq 
    
    rd_exc = np.zeros(( N,N, T ))
    rd_inh = np.zeros(( N, T ))
    
    sigmarange = model.params["sigmarange"]
    ds = model.params["ds"]
    Irange = model.params["Irange"]
    dI = model.params["dI"]
    precalc_r = model.params["precalc_r"]
    precalc_tau_mu = model.params["precalc_tau_mu"]
    precalc_V = model.params["precalc_V"]
    
    interpolate_rate = model.params["interpolate_rate"]
    interpolate_V = model.params["interpolate_V"]
    interpolate_tau = model.params["interpolate_tau"]
    
    print("interpolate adjoint : ", interpolate_rate, interpolate_V, interpolate_tau)
    ##############################################
    
    if (startind_ > 1):
        fo.adjust_shape_init_params(model, init_vars, startind_)
    
    t_pre_ndt = np.around(t_sim_pre_ / dt).astype(int)
    delay_state_vars_ = [ np.zeros(( N, V, startind_ )) ] * noise_real
    
    if ( startind_ > 1 and t_pre_ndt <= startind_ ):
        logging.error("Not possible to set up initial conditions without sufficient simulation time before control")
        #return
        
    state_pre_ = np.zeros(( noise_real, N,V,max(t_pre_ndt+1,1) ))
    state_pre_mean_ = np.zeros(( N,V,max(t_pre_ndt+1,1) ))
    init_vars_sim = [None] * noise_real
    init_vars_post = [None] * noise_real
    
    # simulate with duration t_sim_pre before start
    if (t_sim_pre_ >= dt):
        model.params['duration'] = t_sim_pre_
        control_pre_ = model.getZeroControl()

        for noise_ in range(noise_real):
            fo.set_init_1d(init_params, model)
            state_pre_[noise_,:,:,:] = fo.updateFullState(model, control_pre_, state_vars)
            if startind_ == 1:
                fo.update_init(model, init_vars, state_vars)
            else:
                fo.update_init_delayed(model, delay_state_vars_[noise_], init_vars, state_vars, t_pre_ndt, startind_)

            init_vars_sim[noise_] = fo.get_init(model, init_vars, state_vars, startind_, delay_state_vars_[noise_][:,:,-startind_:])

            state_pre_mean_[:,:,:] +=state_pre_[noise_,:,:,:]

        state_pre_mean_ /= noise_real

        #return

        """     
        if startind_ == 1:
            fo.update_init(model, init_vars, state_vars)
        else:
            fo.update_init_delayed(model, delay_state_vars_, init_vars, state_vars, t_pre_ndt, startind_)
        """

    #print(state_pre_[0][0,0,10], state_pre_[1][0,0,10])
    #print(init_vars_sim[0][0,0,2], init_vars_sim[1][0,0,2])
    #print(init_vars_post[0][0,0,2], init_vars_post[1][0,0,2])
    
    model.params['duration'] = t_sim_

    # state0_ is for noise realization, state1_ is for mean realization
    state0_ = np.zeros(( noise_real, N, V, T ))
    for noise_ in range(noise_real):
        fo.set_init(model, init_vars_sim[noise_], init_vars, state_vars, startind_)
        #print(control_.shape)
        state0_[noise_,:,:,:] = fo.updateFullState(model, control_, state_vars)
    
    state1_ = np.zeros(( noise_real, N, V, T ))
    state_mean_ = np.zeros(( N, V, T ))
    u_opt0_ = control_.copy()
    best_control_ = control_.copy()
    
    # set max control
    best_control_ = fo.setmaxcontrol(n_control_vars, best_control_, cntrl_max_, cntrl_min_)  
    
    ip_, ie_, is_ = cost.getParams()

    # set precision penalty to zero for transition time
    for t_ in range(T):
        if t_ < transition_time_ * T:
            for n_ in range(N):
                for v_ in range(2):
                        target_state_[n_,v_,t_] = - 1000.
                        
    total_cost_ = np.zeros(( noise_real, max_iteration_+1 ))
    total_cost_mean_std = np.zeros(( max_iteration_+1, 2 ))
    for noise_ in range(noise_real):
        total_cost_[noise_,i] = cost.f_int(N, n_control_vars, T, dt, state0_[noise_,:,:,:], target_state_, best_control_,
                                ip_, ie_, is_, v_ = prec_variables )
    
    
    total_cost_mean_std[i,0] = np.mean( total_cost_[:,i] )
    total_cost_mean_std[i,1] = np.std( total_cost_[:,i] )
    

    runtime_ = np.zeros(( int(max_iteration_+1) ))
    runtime_start_ = timer()
    
    #print(total_cost_)
    print("RUN ", i, ", total integrated cost = ", np.mean(total_cost_[:,i]) )
    
    if CGVar not in VALID_VAR:
        print("No valid variant of gradient descent selected, use None instead.")
        CGVar = None
    else:
        print("Gradient descend method: ", CGVar)
    
    full_cost_grad = np.zeros(( N, 2, T ))   
    
    grad0_ = np.zeros(( N, n_control_vars, T ))
    grad1_ = grad0_.copy()
    phi0_ = grad0_.copy()
    dir0_ = grad0_.copy()

    grad_noise_ = np.zeros(( noise_real, N, n_control_vars, T ))
    model.params['duration'] = t_sim_

    start_step_noise_ = startStep_
    startstep_exc_ = startStep_
    startstep_inh_ = startStep_
    startstep_joint_ = startStep_
    
    if line_search_ == None:
        line_search_func = fo.step_size
    elif line_search_ == "AG":
        line_search_func = fo.AG_line_search
    else:
        print("No valid line search selected use bisection instead")
        line_search_func = fo.step_size

    smooth_grad = False
    smooth_t = 100

    abort = False
        
    while( i < max_iteration_ ):

        grad0_ = grad1_.copy()
        grad1_ = np.zeros(( N, n_control_vars, T ))

        state_mean_ = np.zeros(( N, V, T ))

        for noise_ in range(noise_real):
            state_mean_ += state0_[noise_,:,:,:]

        state_mean_ /= noise_real
                
        for ind_time in range(T):
            f_p_grad_t_ = cost.cost_precision_gradient_t(N, V_target, state_mean_[:,:2,ind_time], target_state_[:,:,ind_time], ip_)

            for v in prec_variables:
                full_cost_grad[0,v,ind_time] = f_p_grad_t_[0,v] 

            rd_exc[0,0, ind_time] = state_mean_[0,0,ind_time] * 1e-3        
            rd_inh[0, ind_time] = state_mean_[0,1,ind_time] * 1e-3

        if smooth_grad:
            for ind_time in range(T-1,1,-1):
                for v in prec_variables:
                    if ind_time > smooth_t:
                        full_cost_grad[0,v,ind_time] = np.mean(full_cost_grad[0,v,ind_time-smooth_t:ind_time])
                    else:
                        full_cost_grad[0,v,ind_time] = np.mean(full_cost_grad[0,v,:ind_time])

        #full_cost_grad /= noise_real
            
            #plt.plot(np.arange(T), full_cost_grad[0,0,:])
            #plt.plot(np.arange(T), full_cost_grad[0,1,:])
            #plt.show()

        #for noise_ in range(noise_real):
                                                        
        phi0_ = phi(N, V, T, dt, state_mean_[:,:,:], best_control_, full_cost_grad, state_pre_mean_, 
                    ext_exc_current,
                    ext_inh_current,
                    ext_ee_rate,
                    ext_ei_rate,
                    ext_ie_rate,
                    ext_ii_rate,
                    sigmae_ext,
                    sigmai_ext,
                    a,
                    b,
                    tauA,
                    C,
                    Ke,
                    Ki,
                    tau_se,
                    tau_si,
                    Jee_max,
                    Jei_max,
                    Jie_max,
                    Jii_max,
                    tau_se_sq,
                    tau_si_sq,
                    Jee_sq,
                    Jei_sq,
                    Jie_sq,
                    Jii_sq,
                    taum,
                    tau_ou,
                    ndt_de,
                    ndt_di,
                    factor_ee1,
                    factor_ee2,
                    factor_ei1,
                    factor_ei2,
                    factor_ie1,
                    factor_ie2,
                    factor_ii1,
                    factor_ii2,
                    factor_eec1,
                    factor_eec2,
                    factor_eic1,
                    factor_eic2,
                    factor_iec1,
                    factor_iec2,
                    factor_iic1,
                    factor_iic2,
                    rd_exc,
                    rd_inh,
                    sigmarange, ds, Irange, dI, 
                    precalc_r, precalc_tau_mu, precalc_V,
                    interpolate_rate,
                    interpolate_V,
                    interpolate_tau,
                    )
                
        if ( total_cost_[noise_,i] < tolerance_ ):
            print("Cost negligibly small.")
            max_iteration_ = i
            break                    
            
        phi1_ = phi1(N, V, T, n_control_vars, phi0_, state0_[noise_], best_control_, state_pre_mean_,
                            sigmae_ext,
                            sigmai_ext,
                            ext_ee_rate,
                            ext_ei_rate,
                            ext_ie_rate,
                            ext_ii_rate,
                            tau_se,
                            tau_si,
                            tau_se_sq,
                            tau_si_sq,
                            Jee_sq,
                            Jei_sq,
                            Jie_sq,
                            Jii_sq,
                            taum,
                            factor_ee1,
                            factor_ee2,
                            factor_ei1,
                            factor_ei2,
                            factor_ie1,
                            factor_ie2,
                            factor_ii1,
                            factor_ii2,
                            factor_eec1,
                            factor_eec2,
                            factor_eic1,
                            factor_eic2,
                            factor_iec1,
                            factor_iec2,
                            factor_iic1,
                            factor_iic2,
                            rd_exc,
                            rd_inh,
                            ndt_de,
                            ndt_di,
                        )
        

        grad1_ = fo.compute_gradient(N, n_control_vars, T, dt, best_control_, grad_noise_[noise_], phi1_, control_variables, ie_, is_)

        i += 1   
        dir0_ = fo.set_direction(N, T, n_control_vars, grad0_, grad1_, dir0_, i, CGVar, tolerance_)

        step_, dir0_, cost_ = fo.get_step_noise(line_search_func, model, N, n_control_vars, T, dt, noise_real, state0_, target_state_,
                    best_control_, dir0_, ip_, ie_, is_, startstep_exc_, startstep_inh_, start_step_noise_, cntrl_max_,
                    cntrl_min_, grad1_, total_cost_[:,i-1], init_vars_sim, startind_, prec_vars=prec_variables, control_vars=control_variables, separate_comp=True, noise_in_step_comp = noise_in_step_comp_)
        
        #print("std dev of cost ", np.std(cost_), np.std(cost_)/np.mean(cost_))
        total_cost_[:,i] = cost_
        runtime_[i] = timer() - runtime_start_
        
        if (
            i <= 20
            or ( i <= 200 and i%10 == 0 ) 
            or ( i <= 2000 and i%100 == 0 ) 
            or ( i%1000 == 0 )
            ):
            print( "RUN ", i, ", total integrated cost = ", np.mean( total_cost_[:,i] ) )

        total_cost_mean_std[i,0] = np.mean( total_cost_[:,1:i+1] )
        total_cost_mean_std[i,1] = np.std( total_cost_mean_std[1:i+1,0] )

        #print( total_cost_mean_std[:i+1,0] )
        #print(total_cost_mean_std[i,:])
            
        best_control_ = u_opt0_ + step_ * dir0_
        best_control_ = fo.setmaxcontrol(n_control_vars, best_control_, cntrl_max_, cntrl_min_)                
        
        u_diff_ = ( np.absolute(best_control_ - u_opt0_) < tolerance_ )
        if ( u_diff_.all() ):
            print("Control only changes marginally.")
            max_iteration_ = i
            break
        
        if ( np.amax(np.absolute(grad1_)) < tolerance_ ):
            print("Gradient negligibly small.")
            max_iteration_ = i
            break

        if i > 4 and not abort:
            if ( total_cost_mean_std[i,0] > total_cost_mean_std[i-1,0]
                and total_cost_mean_std[i,0] > total_cost_mean_std[i-2,0]
                and total_cost_mean_std[i,0] > total_cost_mean_std[i-3,0] 
                ):
                print("no cost improvement")
                abort = True
                

        if ( abort and total_cost_mean_std[i,0] < total_cost_mean_std[i-1,0]
            and total_cost_mean_std[i,0] < total_cost_mean_std[i-2,0]):
            max_iteration_ = i
            break
        
        u_opt0_ = best_control_.copy()   
        state_mean_ = np.zeros(( N, V, T ))

        for noise_ in range(noise_real):
            fo.set_init(model, init_vars_sim[noise_], init_vars, state_vars, startind_)
            state1_[noise_,:,:,:] = fo.updateFullState(model, best_control_, state_vars)

            state_mean_[:,:,:] += state1_[noise_,:,:,:]
        
        state_mean_ /= noise_real
        state0_ = state1_.copy() 

        if (t_sim_pre_ >= dt):
            model.params['duration'] = t_sim_pre_
            control_pre_ = model.getZeroControl()

            for noise_ in range(noise_real):
                fo.set_init_1d(init_params, model)
                state_pre_[noise_] = fo.updateFullState(model, control_pre_, state_vars)
                if startind_ == 1:
                    fo.update_init(model, init_vars, state_vars)
                else:
                    fo.update_init_delayed(model, delay_state_vars_[noise_], init_vars, state_vars, t_pre_ndt, startind_)

                init_vars_sim[noise_] = fo.get_init(model, init_vars, state_vars, startind_, delay_state_vars_[noise_][:,:,-startind_:])
    
        model.params['duration'] = t_sim_

    for noise_ in range(noise_real): 
        fo.set_init(model, init_vars_sim[noise_], init_vars, state_vars, startind_)     
        state1_[noise_,:,:,:] = fo.updateFullState(model, best_control_, state_vars)
        if startind_ == 1:
            fo.update_init(model, init_vars, state_vars)
        else:
            fo.update_init_delayed(model, delay_state_vars_[noise_], init_vars, state_vars, T, startind_)

        init_vars_post[noise_] = fo.get_init(model, init_vars, state_vars, startind_, delay_state_vars_[noise_][:,:,-startind_:])
    
        for j in [5,6,7,8]:
            if np.amin(state1_[noise_][0,j,:]) < 0. or np.amax(state1_[noise_][0,j,:]) > 1.:
                print("WARNING: s-parameter not in proper range")
    
    improvement = 100.
    if np.mean(total_cost_[:,0]) != 0.:
        improvement = 100. - ( 100. *( np.mean(total_cost_[:,max_iteration_]) / np.mean(total_cost_[:,0]) ) )
        
    print("RUN ", max_iteration_, ", total integrated cost mean = ", np.mean(total_cost_[:,max_iteration_]))
    print("Improved over ", max_iteration_, " iterations in ", runtime_[max_iteration_]," seconds by ", improvement, " percent.")
    
    # compute node-wise cost in precision, energy, sparsity for weight = 1
    cost_node_ = cost.cost_int_per_node(N, n_control_vars, T, dt, state_mean_, target_state_,
                                     best_control_, 1., 1., 1., v_ = prec_variables )

    cost_node = [ np.zeros(( N, 2 )), np.zeros(( N, 6 )), np.zeros(( N, 6 )) ]

    for j1 in range(2):
        cost_node[0][0,j1] = cost_node_[0][0,j1]
    for j1 in range(6):
        cost_node[1][0,j1] = cost_node_[1][0,j1]
    for j1 in range(6):
        cost_node[2][0,j1] = cost_node_[2][0,j1]
        
    if (t_sim_pre_ < dt and t_sim_post_ < dt):
        return best_control_, state1_, total_cost_, runtime_, grad1_, phi0_, cost_node, total_cost_mean_std
    
    t_post_ndt = np.around(t_sim_post_ / dt).astype(int)
    state_post_ = np.zeros(( noise_real, N,V,t_post_ndt+1))
    state_post_mean_ = np.zeros(( N,V,t_post_ndt+1))
    
    if (t_sim_post_ >= dt): 
        model.params.duration = t_sim_post_ #- dt
        control_post_ = model.getZeroControl()
        for noise_ in range(noise_real):
            fo.set_init(model, init_vars_post[noise_], init_vars, state_vars, startind_)
            state_post_[noise_,:,:,:] = fo.updateFullState(model, control_post_, state_vars)
            state_post_mean_ += state_post_[noise_,:,:,:]
        state_post_mean_ /= noise_real
 
    model.params.duration = t_sim_ + t_sim_pre_ + t_sim_post_
    bc_ = model.getZeroControl()
    bs_ = [model.getZeroFullState()] * noise_real
        
    i1 = int(round(t_sim_pre_/dt, 1))
    i2 = int(round(t_sim_post_/dt, 1))
    
    for noise_ in range(noise_real):
        fo.set_pre_post(i1, i2, bc_, bs_[noise_], best_control_, state_pre_[noise_,:,:,:], state1_[noise_,:,:,:], state_post_[noise_,:,:,:], state_vars, model.params.a, model.params.b)
            
    return bc_, bs_, total_cost_, runtime_, grad1_, phi0_, cost_node, total_cost_mean_std

@numba.njit
def phi(N, V, T, dt, state_, control_, full_cost_grad, state_pre_,
                    ext_exc_current,
                    ext_inh_current,
                    ext_ee_rate,
                    ext_ei_rate,
                    ext_ie_rate,
                    ext_ii_rate,
                    sigmae_ext,
                    sigmai_ext,
                    a,
                    b,
                    tauA,
                    C,
                    Ke,
                    Ki,
                    tau_se,
                    tau_si,
                    Jee_max,
                    Jei_max,
                    Jie_max,
                    Jii_max,
                    tau_se_sq,
                    tau_si_sq,
                    Jee_sq,
                    Jei_sq,
                    Jie_sq,
                    Jii_sq,
                    taum,
                    tau_ou,
                    ndt_de,
                    ndt_di,
                    factor_ee1,
                    factor_ee2,
                    factor_ei1,
                    factor_ei2,
                    factor_ie1,
                    factor_ie2,
                    factor_ii1,
                    factor_ii2,
                    factor_eec1,
                    factor_eec2,
                    factor_eic1,
                    factor_eic2,
                    factor_iec1,
                    factor_iec2,
                    factor_iic1,
                    factor_iic2,
                    rd_exc,
                    rd_inh,
                    sigmarange, ds, Irange, dI, 
                    precalc_r, precalc_tau_mu, precalc_V,
                    interpolate_rate,
                    interpolate_V,
                    interpolate_tau,
                    ):
    
    phi_ = np.zeros(( N, V, T ))
    
    jac = jacobian(V, state_[:,:,:], control_[:,:,:], T, state_pre_,
                       ext_exc_current,
                       ext_inh_current,
                       ext_ee_rate,
                       ext_ei_rate,
                       ext_ie_rate,
                       ext_ii_rate,
                       sigmae_ext,
                       sigmai_ext,
                       a,
                       b,
                       tauA,
                       tau_se,
                       tau_si,
                       Jee_max,
                       Jei_max,
                       Jie_max,
                       Jii_max,
                       tau_se_sq,
                       tau_si_sq,
                       Jee_sq,
                       Jei_sq,
                       Jie_sq,
                       Jii_sq,
                       taum,
                       tau_ou,
                       factor_ee1,
                       factor_ee2,
                       factor_ei1,
                       factor_ei2,
                       factor_ie1,
                       factor_ie2,
                       factor_ii1,
                       factor_ii2,
                       factor_eec1,
                       factor_eec2,
                       factor_eic1,
                       factor_eic2,
                       factor_iec1,
                       factor_iec2,
                       factor_iic1,
                       factor_iic2,
                       rd_exc,
                       rd_inh,
                       sigmarange, ds, Irange, dI,
                       C,
                       precalc_r, precalc_tau_mu, precalc_V,
                       ndt_de,
                       ndt_di,
                       interpolate_rate,
                       interpolate_V,
                       interpolate_tau,
                       ) 
    
    #print("n_dt de = ", ndt_de)
    #print("jac 7 = ", jac[7,0,900:])
    #print("jac 2 = ", jac[0,2,400:])
            
    for ind_time in range(T-1, -1, -1):
        
        if ind_time + ndt_de >= T-1:
            phi_[0,0,ind_time] = - ( full_cost_grad[0,0,ind_time]
                                    + np.dot( np.array( [ phi_[0,4,ind_time] ] ),
                                             np.array( [ jac[4,0,ind_time] ] ) ) )
        elif ind_time + ndt_de + 1 >= T-1:
            phi_[0,0,ind_time] = - ( full_cost_grad[0,0,ind_time]
                                    + np.dot( np.array( [ phi_[0,4,ind_time],
                                                         phi_[0,15,ind_time+ndt_de],
                                                         phi_[0,16,ind_time+ndt_de] ] ),
                                             np.array( [ jac[4,0,ind_time], 
                                                        jac[15,0,ind_time+ndt_de],
                                                        jac[16,0,ind_time+ndt_de] ] ) ) )
        else:
            phi_[0,0,ind_time] = - ( full_cost_grad[0,0,ind_time]
                                    + np.dot( np.array( [ phi_[0,4,ind_time],
                                                         phi_[0,5,ind_time+ndt_de+1],
                                                         phi_[0,7,ind_time+ndt_de+1],
                                                         phi_[0,9,ind_time+ndt_de+1],
                                                         phi_[0,11,ind_time+ndt_de+1],
                                                         phi_[0,15,ind_time+ndt_de],
                                                         phi_[0,16,ind_time+ndt_de] ] ),
                                             np.array( [ jac[4,0,ind_time], 
                                                        jac[5,0,ind_time+ndt_de],  # should not be plus 1
                                                        jac[7,0,ind_time+ndt_de],
                                                        jac[9,0,ind_time+ndt_de],
                                                        jac[11,0,ind_time+ndt_de],
                                                        jac[15,0,ind_time+ndt_de],
                                                        jac[16,0,ind_time+ndt_de] ] ) ) )
            
        if ind_time + ndt_di >= T-1:
            phi_[0,1,ind_time] = - ( full_cost_grad[0,1,ind_time] )
        elif ind_time + ndt_di + 1 >= T-1:
            phi_[0,1,ind_time] = - ( full_cost_grad[0,1,ind_time]
                                    + np.dot( np.array( [phi_[0,15,ind_time+ndt_di],
                                                         phi_[0,16,ind_time+ndt_di] ] ),
                                             np.array( [ jac[15,1,ind_time+ndt_di],
                                                        jac[16,1,ind_time+ndt_di] ] ) ) )
        else:
            phi_[0,1,ind_time] = - ( full_cost_grad[0,1,ind_time]
                                    + np.dot( np.array( [ phi_[0,6,ind_time+ndt_di+1],
                                                         phi_[0,8,ind_time+ndt_di+1],
                                                         phi_[0,10,ind_time+ndt_di+1],
                                                         phi_[0,12,ind_time+ndt_di+1],
                                                         phi_[0,15,ind_time+ndt_di],
                                                         phi_[0,16,ind_time+ndt_di] ] ),
                                             np.array( [ jac[6,1,ind_time+ndt_di],
                                                        jac[8,1,ind_time+ndt_di],
                                                        jac[10,1,ind_time+ndt_di],
                                                        jac[12,1,ind_time+ndt_di],
                                                        jac[15,1,ind_time+ndt_di],
                                                        jac[16,1,ind_time+ndt_di] ] ) ) )         
            
        
        
        #if ind_time == T-1:
        #    phi_[0,9,-1] = + dt * jac[15,9,-1] * phi_[0,0,-1] * jac[0,15,-1]
        #    print("last entry in phi 9 = ", phi_[0,9,-1])
        if ind_time == 0:
            break
        
        res = - phi_[0,2,ind_time] * jac[2,18, ind_time-1]
        phi_[0,18,ind_time-1] = res
        
        res = - phi_[0,3,ind_time] * jac[3,19, ind_time-1]
        phi_[0,19,ind_time-1] = res
            
        der = 0.
        der += phi_[0,0,ind_time] * jac[0,2,ind_time-1]
        
        # if not static state before simulation, should go back further in time
        
        # 2,2,-1 checked by trial and error
        der += phi_[0,2,ind_time] * jac[2,2,ind_time-1]
        der += phi_[0,17,ind_time] * jac[17,2,ind_time-1]
        der += phi_[0,18,ind_time-1] * jac[18,2,ind_time-1]
        phi_[0,2,ind_time-1] = phi_[0,2,ind_time] - dt * der

        der = 0.
        der += phi_[0,1,ind_time] * jac[1,3,ind_time-1]
        der += ( phi_[0,3,ind_time] * jac[3,3,ind_time-1] )
        der += phi_[0,19,ind_time-1] * jac[19,3,ind_time-1]
        phi_[0,3,ind_time-1] = phi_[0,3,ind_time] - dt * der
        
        der = ( phi_[0,0,ind_time] * jac[0,4,ind_time-1]
               + phi_[0,4,ind_time] * jac[4,4,ind_time-1]
               + phi_[0,17,ind_time] * jac[17,4,ind_time-1]
               + phi_[0,18,ind_time-1] * jac[18,4,ind_time-1] )
        phi_[0,4,ind_time-1] = phi_[0,4,ind_time] - dt * der
        
        der = ( phi_[0,2,ind_time] * jac[2,5,ind_time-1]
               + phi_[0,5,ind_time] * jac[5,5,ind_time-1]
               + phi_[0,9,ind_time] * jac[9,5,ind_time-1]
               )
        phi_[0,5,ind_time-1] = phi_[0,5,ind_time] - dt * der
        
        der = ( phi_[0,2,ind_time] * jac[2,6,ind_time-1]
               + phi_[0,6,ind_time] * jac[6,6,ind_time-1]
               + phi_[0,10,ind_time] * jac[10,6,ind_time-1]
               )
        phi_[0,6,ind_time-1] = phi_[0,6,ind_time] - dt * der
        
        der = ( phi_[0,3,ind_time] * jac[3,7,ind_time-1]
               + phi_[0,7,ind_time] * jac[7,7,ind_time-1]
               + phi_[0,11,ind_time] * jac[11,7,ind_time-1]
               )
        phi_[0,7,ind_time-1] = phi_[0,7,ind_time] - dt * der
        
        der = ( phi_[0,3,ind_time] * jac[3,8,ind_time-1]
               + phi_[0,8,ind_time] * jac[8,8,ind_time-1]
               + phi_[0,12,ind_time] * jac[12,8,ind_time-1]
               )
        phi_[0,8,ind_time-1] = phi_[0,8,ind_time] - dt * der

        """
        ## adjoint state of ou noise does not enter computation later, can be neglected
        der = ( phi_[0,2,ind_time] * jac[2,13,ind_time-1]
               + phi_[0,13,ind_time] * jac[13,13,ind_time-1]
               )
        phi_[0,13,ind_time-1] = phi_[0,13,ind_time] - dt * der

        der = ( phi_[0,3,ind_time] * jac[3,14,ind_time-1]
               + phi_[0,14,ind_time] * jac[14,14,ind_time-1]
               )
        phi_[0,14,ind_time-1] = phi_[0,14,ind_time] - dt * der
        """
        
        
        #der = ( phi_[0,16,ind_time] * jac[16,11,ind_time] )
        #phi_[0,11,ind_time-1] = phi_[0,11,ind_time] - dt * der
        
        res = ( - phi_[0,0,ind_time] * jac[0,15, ind_time-1]
               - phi_[0,17,ind_time] * jac[17,15, ind_time-1]
               - phi_[0,18,ind_time-1] * jac[18,15, ind_time-1] )
        phi_[0,15,ind_time-1] = res
        
        res = - phi_[0,1,ind_time] * jac[1,16, ind_time-1] - phi_[0,19,ind_time-1] * jac[19,16, ind_time-1]
        phi_[0,16,ind_time-1] = res
        
        res = - phi_[0,4,ind_time-1] * jac[4,17, ind_time-1]
        phi_[0,17,ind_time-1] = res
        
        der = ( phi_[0,9,ind_time] * jac[9,9,ind_time-1] + phi_[0,15,ind_time-1] * jac[15,9,ind_time-1] )
        phi_[0,9,ind_time-1] = phi_[0,9,ind_time] - dt * der
        
        der = ( phi_[0,10,ind_time] * jac[10,10,ind_time-1] + phi_[0,15,ind_time-1] * jac[15,10,ind_time-1] )
        phi_[0,10,ind_time-1] = phi_[0,10,ind_time] - dt * der
        
        der = ( phi_[0,11,ind_time] * jac[11,11,ind_time-1] + phi_[0,16,ind_time-1] * jac[16,11,ind_time-1] )
        phi_[0,11,ind_time-1] = phi_[0,11,ind_time] - dt * der
        
        der = ( phi_[0,12,ind_time] * jac[12,12,ind_time-1] + phi_[0,16,ind_time-1] * jac[16,12,ind_time-1] )
        phi_[0,12,ind_time-1] = phi_[0,12,ind_time] - dt * der
    
    
                
    return phi_

@numba.njit
def phi1(N, V, T, n_control_vars, phi_, state_, control_, state_pre_,
                          sigmae_ext,
                          sigmai_ext,
                          ext_ee_rate,
                          ext_ei_rate,
                          ext_ie_rate,
                          ext_ii_rate,
                          tau_se,
                          tau_si,
                          tau_se_sq,
                          tau_si_sq,
                          Jee_sq,
                          Jei_sq,
                          Jie_sq,
                          Jii_sq,
                          taum,
                          factor_ee1,
                          factor_ee2,
                          factor_ei1,
                          factor_ei2,
                          factor_ie1,
                          factor_ie2,
                          factor_ii1,
                          factor_ii2,
                          factor_eec1,
                          factor_eec2,
                          factor_eic1,
                          factor_eic2,
                          factor_iec1,
                          factor_iec2,
                          factor_iic1,
                          factor_iic2,
                          rd_exc,
                          rd_inh,
                          ndt_de,
                          ndt_di,
                         ):  
    
    phi1_ = np.zeros(( N, n_control_vars, T ))
            
        # could leave shift at zero in jacobian and algorithm would do almost equally well
            
    for ind_t in range(0, T):
        
        jac_u_ = D_u_h(V, n_control_vars, state_, control_, ind_t, state_pre_,
                      sigmae_ext,
                      sigmai_ext,
                      ext_ee_rate,
                      ext_ei_rate,
                      ext_ie_rate,
                      ext_ii_rate,
                      tau_se,
                      tau_si,
                      tau_se_sq,
                      tau_si_sq,
                      Jee_sq,
                      Jei_sq,
                      Jie_sq,
                      Jii_sq,
                      taum,
                      factor_ee1,
                      factor_ee2,
                      factor_ei1,
                      factor_ei2,
                      factor_ie1,
                      factor_ie2,
                      factor_ii1,
                      factor_ii2,
                      factor_eec1,
                      factor_eec2,
                      factor_eic1,
                      factor_eic2,
                      factor_iec1,
                      factor_iec2,
                      factor_iic1,
                      factor_iic2,
                      rd_exc,
                      rd_inh,
                      ndt_de,
                      ndt_di,
                    )
                
        phi = np.ascontiguousarray(phi_[0,:,ind_t])
        #phi_shift = np.ascontiguousarray(phi_[0,:,ind_t-1])#, dtype=np.float64)
        #phi_shift = np.ascontiguousarray(phi_[0,:,ind_t])
        phi_shift = phi.copy()
        

        if ind_t == T-1:
            for i in [5,6,7,8,9,10,11,12]:
                phi_shift[i] = 0.
        else:
            for i in [5,6,7,8,9,10,11,12]:
                phi_shift[i] = phi_[0,i,ind_t+1]
           
        #print("t, phi 9, 15 = ", ind_t, phi_shift[9], phi_shift[15])
             
        y0 = np.ascontiguousarray(jac_u_[0,:])
        y1 = np.ascontiguousarray(jac_u_[1,:])
        y2 = np.ascontiguousarray(jac_u_[2,:])
        y3 = np.ascontiguousarray(jac_u_[3,:])
        y4 = np.ascontiguousarray(jac_u_[4,:])
        y5 = np.ascontiguousarray(jac_u_[5,:])
        
        phi1_[0,2,ind_t] = np.dot(phi_shift, y2)
        phi1_[0,3,ind_t] = np.dot(phi_shift, y3)
        phi1_[0,4,ind_t] = np.dot(phi_shift, y4)
        phi1_[0,5,ind_t] = np.dot(phi_shift, y5)
        
        #print("time, phi 6,10 15, y3 = ", ind_t, phi_shift[6], phi_shift[10], phi_shift[15], y3)

        if ind_t > 0:
            phi1_[0,0,ind_t] = np.dot(phi_shift, y0)
            phi1_[0,1,ind_t] = np.dot(phi_shift, y1)
        
        
        #print(phi1_[0,0,ind_t] == res1, phi1_[0,1,ind_t] == res2)

    return phi1_

@numba.njit
def D_xdot(V, state_t_):
    dxdot_ = np.zeros(( V, V ))
    return dxdot_

@numba.njit
def D_u_h(V, n_control_vars, state_, control_, t_, state_pre_,
          sigmae_ext,
          sigmai_ext,
          ext_ee_rate,
          ext_ei_rate,
          ext_ie_rate,
          ext_ii_rate,
          tau_se,
          tau_si,
          tau_se_sq,
          tau_si_sq,
          Jee_sq,
          Jei_sq,
          Jie_sq,
          Jii_sq,
          taum,
          factor_ee1,
          factor_ee2,
          factor_ei1,
          factor_ei2,
          factor_ie1,
          factor_ie2,
          factor_ii1,
          factor_ii2,
          factor_eec1,
          factor_eec2,
          factor_eic1,
          factor_eic2,
          factor_iec1,
          factor_iec2,
          factor_iic1,
          factor_iic2,
          rd_exc,
          rd_inh,
          shift_e,
          shift_i,
          ):
    
    if t_-shift_e >= 0:
        z1ee = factor_ee1 * rd_exc[0,0,t_-shift_e] + factor_eec1 * ( control_[0,2,t_] + ext_ee_rate )
        z1ie = factor_ie1 * rd_exc[0,0,t_-shift_e] + factor_iec1 * ( control_[0,4,t_] + ext_ie_rate )
    else:
        z1ee = factor_ee1 * state_pre_[0,0,t_-shift_e-1] * 1e-3 + factor_eec1 * ( control_[0,2,t_] + ext_ee_rate )
        z1ie = factor_ie1 * state_pre_[0,0,t_-shift_e-1] * 1e-3 + factor_iec1 * ( control_[0,4,t_] + ext_ie_rate )
        
    if t_-shift_i >= 0:
        z1ei = factor_ei1 * rd_inh[0,t_-shift_i] + factor_eic1 * ( control_[0,3,t_] + ext_ei_rate )
        z1ii = factor_ii1 * rd_inh[0,t_-shift_i] + factor_iic1 * ( control_[0,5,t_] + ext_ii_rate )
    else:
        z1ei = factor_ei1 * state_pre_[0,1,t_-shift_i-1] * 1e-3 + factor_eic1 * ( control_[0,3,t_] + ext_ei_rate )
        z1ii = factor_ii1 * state_pre_[0,1,t_-shift_i-1] * 1e-3 + factor_iic1 * ( control_[0,5,t_] + ext_ii_rate )

    #z1ee = max(z1ee,0.)
    #z2ee = max(z2ee,0.)
    
    duh_ = np.zeros(( n_control_vars, V ))
    
    # -1 checked by trial and error
    duh_[0,2] = - 1. / state_[0,18,t_-1]
    duh_[1,3] = - 1. / state_[0,19,t_-1]
    
    # ee, ei, ie, ii
    duh_[2,5] = - factor_eec1 * (1. - state_[0,5,t_]) / tau_se
    duh_[2,9] = ( - ( 1. - state_[0,5,t_] )**2 * factor_eec2
                 - ( factor_eec2 - 2. * tau_se * factor_eec1 ) * state_[0,9,t_] ) / tau_se_sq 
    
    duh_[3,6] = - factor_eic1 * (1. - state_[0,6,t_]) / tau_si
    duh_[3,10] = ( - ( 1. - state_[0,6,t_] )**2 * factor_eic2
                 - ( factor_eic2 - 2. * tau_si * factor_eic1 ) * state_[0,10,t_] ) / tau_si_sq 
    
    duh_[4,7] = - factor_iec1 * (1. - state_[0,7,t_]) / tau_se
    duh_[4,11] = ( - ( 1. - state_[0,7,t_] )**2 * factor_iec2
                 - ( factor_iec2 - 2. * tau_se * factor_iec1 ) * state_[0,11,t_] ) / tau_se_sq
    
    duh_[5,8] = - factor_iic1 * (1. - state_[0,8,t_]) / tau_si
    duh_[5,12] = ( - ( 1. - state_[0,8,t_] )**2 * factor_iic2
                 - ( factor_iic2 - 2. * tau_si * factor_iic1 ) * state_[0,12,t_] ) / tau_si_sq
    
    sigma_ee = 2. * Jee_sq * state_[0,9,t_] * tau_se * taum / ((1. + z1ee) * taum + tau_se)
    sigma_ei = 2. * Jei_sq * state_[0,10,t_] * tau_si * taum * ( (1. + z1ei ) * taum + tau_si )**(-1)
        
    arg = ( sigma_ee + sigma_ei + sigmae_ext**2 )
    if arg > 0:
        sigma_sqrt_e = arg**(-1./2.)
    else:
        sigma_sqrt_e = 0.
        #print("WARNING: sigma e smaller zero")
    
    duh_[2,15] = ( 0.5 * ( (1. + z1ee) * taum + tau_se )**(-2.) * factor_eec1 * taum *
                  ( 2. * Jee_sq * tau_se * taum * state_[0,9,t_] ) * sigma_sqrt_e )
    
    duh_[3,15] = ( 0.5 * ( (1. + z1ei) * taum + tau_si )**(-2.) * factor_eic1 * taum *
                  ( 2. * Jei_sq * tau_si * taum * state_[0,10,t_] ) * sigma_sqrt_e )
    
    sigma_ie = 2. * Jie_sq * state_[0,11,t_] * tau_se * taum / ((1. + z1ie) * taum + tau_se)
    sigma_ii = 2. * Jii_sq * state_[0,12,t_] * tau_si * taum * ( (1. + z1ii ) * taum + tau_si )**(-1)
        
    arg = ( sigma_ie + sigma_ii + sigmai_ext**2 )
    if arg > 0:
        sigma_sqrt_i = arg**(-1./2.)
    else:
        sigma_sqrt_i = 0.
        #print("WARNING: sigma e smaller zero")
    
    duh_[4,16] = ( 0.5 * ( (1. + z1ie) * taum + tau_se )**(-2.) * factor_iec1 * taum *
                  ( 2. * Jie_sq * tau_se * taum * state_[0,11,t_] ) * sigma_sqrt_i )
    
    duh_[5,16] = ( 0.5 * ( (1. + z1ii) * taum + tau_si )**(-2.) * factor_iic1 * taum *
                  ( 2. * Jii_sq * tau_si * taum * state_[0,12,t_] ) * sigma_sqrt_i )
    
    return duh_

@numba.njit
def jacobian(V, state_, control_, T, state_pre_,
              ext_exc_current,
              ext_inh_current,
              ext_ee_rate,
              ext_ei_rate,
              ext_ie_rate,
              ext_ii_rate,
              sigmae_ext,
              sigmai_ext,
              a,
              b,
              tauA,
              tau_se,
              tau_si,
              Jee_max,
              Jei_max,
              Jie_max,
              Jii_max,
              tau_se_sq,
              tau_si_sq,
              Jee_sq,
              Jei_sq,
              Jie_sq,
              Jii_sq,
              taum,
              tau_ou,
              factor_ee1,
              factor_ee2,
              factor_ei1,
              factor_ei2,
              factor_ie1,
              factor_ie2,
              factor_ii1,
              factor_ii2,
              factor_eec1,
              factor_eec2,
              factor_eic1,
              factor_eic2,
              factor_iec1,
              factor_iec2,
              factor_iic1,
              factor_iic2,
              rd_exc,
              rd_inh,
              sigmarange, ds, Irange, dI,
              C,
              precalc_r, precalc_tau_mu, precalc_V,
              shift_e,
              shift_i,  
              interpolate_rate,
              interpolate_V,
              interpolate_tau,
              ):
    
    jacobian_ = np.zeros(( V, V, T))
    
    for t_ in range(0,T):
           
    # ee, ei, ie, ii
        if t_-shift_e >= 0:
            z1ee = factor_ee1 * rd_exc[0,0,t_-shift_e] + factor_eec1 * ( control_[0,2,t_] + ext_ee_rate )
            z2ee = factor_ee2 * rd_exc[0,0,t_-shift_e] + factor_eec2 * ( control_[0,2,t_] + ext_ee_rate ) 
            z1ie = factor_ie1 * rd_exc[0,0,t_-shift_e] + factor_iec1 * ( control_[0,4,t_] + ext_ie_rate )
            z2ie = factor_ie2 * rd_exc[0,0,t_-shift_e] + factor_iec2 * ( control_[0,4,t_] + ext_ie_rate )
        else:
            z1ee = factor_ee1 * state_pre_[0,0,t_-shift_e-1] * 1e-3 + factor_eec1 * ( control_[0,2,t_] + ext_ee_rate )
            z2ee = factor_ee2 * state_pre_[0,0,t_-shift_e-1] * 1e-3 + factor_eec2 * ( control_[0,2,t_] + ext_ee_rate )
            z1ie = factor_ie1 * state_pre_[0,0,t_-shift_e-1] * 1e-3 + factor_iec1 * ( control_[0,4,t_] + ext_ie_rate )
            z2ie = factor_ie2 * state_pre_[0,0,t_-shift_e-1] * 1e-3 + factor_iec2 * ( control_[0,4,t_] + ext_ie_rate )
            
        if t_-shift_i >= 0:
            z1ei = factor_ei1 * rd_inh[0,t_-shift_i] + factor_eic1 * ( control_[0,3,t_] + ext_ei_rate )
            z2ei = factor_ei2 * rd_inh[0,t_-shift_i] + factor_eic2 * ( control_[0,3,t_] + ext_ei_rate )       
            z1ii = factor_ii1 * rd_inh[0,t_-shift_i] + factor_iic1 * ( control_[0,5,t_] + ext_ii_rate )
            z2ii = factor_ii2 * rd_inh[0,t_-shift_i] + factor_iic2 * ( control_[0,5,t_] + ext_ii_rate )
        else:
            z1ei = factor_ei1 * state_pre_[0,1,t_-shift_i-1] * 1e-3 + factor_eic1 * ( control_[0,3,t_] + ext_ei_rate )
            z2ei = factor_ei2 * state_pre_[0,1,t_-shift_i-1] * 1e-3 + factor_eic2 * ( control_[0,3,t_] + ext_ei_rate )         
            z1ii = factor_ii1 * state_pre_[0,1,t_-shift_i-1] * 1e-3 + factor_iic1 * ( control_[0,5,t_] + ext_ii_rate )
            z2ii = factor_ii2 * state_pre_[0,1,t_-shift_i-1] * 1e-3 + factor_iic2 * ( control_[0,5,t_] + ext_ii_rate )
         
        """
        z1ee = max(z1ee,0.)
        z2ee = max(z2ee,0.)
        z1ei = max(z1ei,0.)
        z2ei = max(z2ei,0.)
        z1ie = max(z1ie,0.)
        z2ie = max(z2ie,0.)
        z1ii = max(z1ii,0.)
        z2ii = max(z2ii,0.)
        """

        ## jac[0,0], jac[1,1], jac[15,15], jac[16,16], jac[17,17], jac[18,18], jac[19,19] not missing, but taken into account in the way phi is calculated
        
        jacobian_[0,2,t_] = - d_r_func_mu(state_[0,2,t_] - state_[0,4,t_]/C, sigmarange, ds,
                                          state_[0,15,t_], Irange, dI, C, precalc_r, interpolate_rate) * 1e3
        jacobian_[0,4,t_] = d_r_func_mu(state_[0,2,t_] - state_[0,4,t_]/C, sigmarange, ds,
                                          state_[0,15,t_], Irange, dI, C, precalc_r, interpolate_rate) * 1e3 / C
        jacobian_[0,15,t_] = - d_r_func_sigma(state_[0,2,t_] - state_[0,4,t_]/C, sigmarange, ds,
                                              state_[0,15,t_], Irange, dI, C, precalc_r, interpolate_rate) * 1e3
        
        jacobian_[1,3,t_] = - d_r_func_mu(state_[0,3,t_], sigmarange, ds,
                                          state_[0,16,t_], Irange, dI, C, precalc_r, interpolate_rate) * 1e3
        jacobian_[1,16,t_] = - d_r_func_sigma(state_[0,3,t_], sigmarange, ds,
                                              state_[0,16,t_], Irange, dI, C, precalc_r, interpolate_rate) * 1e3
        
        jacobian_[2,2,t_] = 1. / state_[0,18,t_]
        jacobian_[2,5,t_] = - Jee_max / state_[0,18,t_]
        jacobian_[2,6,t_] = - Jei_max / state_[0,18,t_]
        if t_ < T - 1:
            jacobian_[2,18,t_] = ( Jee_max * state_[0,5,t_] + Jei_max * state_[0,6,t_] + control_[0,0,t_+1] + ext_exc_current
                           + state_[0,13,t_] - state_[0,2,t_] ) / state_[0,18,t_]**2
        else:
            jacobian_[2,18,t_] = ( Jee_max * state_[0,5,t_] + Jei_max * state_[0,6,t_] + ext_exc_current
                           + state_[0,13,t_] - state_[0,2,t_] ) / state_[0,18,t_]**2
        
        jacobian_[3,3,t_] = 1. / state_[0,19,t_]
        jacobian_[3,7,t_] = - Jie_max / state_[0,19,t_]
        jacobian_[3,8,t_] = - Jii_max / state_[0,19,t_]
        if t_ < T - 1:
            jacobian_[3,19,t_] = ( Jie_max * state_[0,7,t_] + Jii_max * state_[0,8,t_] + control_[0,1,t_+1] + ext_inh_current
                           + state_[0,14,t_] - state_[0,3,t_] ) / state_[0,19,t_]**2
        else:
            jacobian_[3,19,t_] = ( Jie_max * state_[0,7,t_] + Jii_max * state_[0,8,t_] + ext_inh_current
                           + state_[0,14,t_] - state_[0,3,t_] ) / state_[0,19,t_]**2
        
        jacobian_[4,0,t_] = - b * 1e-3
        jacobian_[4,4,t_] = 1. / tauA
        jacobian_[4,17,t_] = - a / tauA
                
        jacobian_[5,0,t_] = - factor_ee1 * 1e-3 * ( 1. - state_[0,5,t_] ) / tau_se
        jacobian_[5,5,t_] = ( 1. + z1ee ) / tau_se
        
        jacobian_[6,1,t_] = - factor_ei1 * 1e-3 * ( 1. - state_[0,6,t_] ) / tau_si
        jacobian_[6,6,t_] = ( 1. + z1ei ) / tau_si
        
        jacobian_[7,0,t_] = - factor_ie1 * 1e-3 * ( 1. - state_[0,7,t_] ) / tau_se
        jacobian_[7,7,t_] = ( 1. + z1ie ) / tau_se
        
        jacobian_[8,1,t_] = - factor_ii1 * 1e-3 * ( 1. - state_[0,8,t_] ) / tau_si
        jacobian_[8,8,t_] = ( 1. + z1ii ) / tau_si
        
        jacobian_[9,0,t_] = ( - ( 1. - state_[0,5,t_] )**2 * factor_ee2 * 1e-3
                             - ( factor_ee2 - 2. * tau_se * factor_ee1 ) * 1e-3 * state_[0,9,t_] ) / tau_se_sq
        jacobian_[9,5,t_] = 2. * ( 1. - state_[0,5,t_] ) * z2ee / tau_se_sq
        jacobian_[9,9,t_] = - ( z2ee - 2. * tau_se * ( z1ee + 1 ) ) / tau_se_sq
        
        jacobian_[10,1,t_] = ( - ( 1. - state_[0,6,t_] )**2 * factor_ei2 * 1e-3
                             - ( factor_ei2 - 2. * tau_si * factor_ei1 ) * 1e-3 * state_[0,10,t_] ) / tau_si_sq
        jacobian_[10,6,t_] = 2. * ( 1. - state_[0,6,t_] ) * z2ei / tau_si_sq
        jacobian_[10,10,t_] = - ( z2ei - 2. * tau_si * ( z1ei + 1 ) ) / tau_si_sq
        
        jacobian_[11,0,t_] = ( - ( 1. - state_[0,7,t_] )**2 * factor_ie2 * 1e-3
                              - ( factor_ie2 - 2. * tau_se * factor_ie1 ) * 1e-3 * state_[0,11,t_] ) / tau_se_sq
        jacobian_[11,7,t_] = 2. * ( 1. - state_[0,7,t_] ) * z2ie / tau_se_sq
        jacobian_[11,11,t_] = - ( z2ie - 2. * tau_se * ( z1ie + 1 ) ) / tau_se_sq
        
        jacobian_[12,1,t_] = ( - ( 1. - state_[0,8,t_] )**2 * factor_ii2 * 1e-3
                             - ( factor_ii2 - 2. * tau_si * factor_ii1 ) * 1e-3 * state_[0,12,t_]) / tau_si_sq
        jacobian_[12,8,t_] = 2. * ( 1. - state_[0,8,t_] ) * z2ii / tau_si_sq
        jacobian_[12,12,t_] = - ( z2ii - 2. * tau_si * ( z1ii + 1 ) ) / tau_si_sq
        
        sigma_ee = 2. * Jee_sq * state_[0,9,t_] * tau_se * taum / ((1. + z1ee) * taum + tau_se)
        sigma_ei = 2. * Jei_sq * state_[0,10,t_] * tau_si * taum * ( (1. + z1ei ) * taum + tau_si )**(-1)
        
        arg = ( sigma_ee + sigma_ei + sigmae_ext**2 )
        if arg > 0:
            sigma_sqrt_e = arg**(-1./2.)
        else:
            sigma_sqrt_e = 0.
            #print("WARNING: sigma e smaller zero")
        
        jacobian_[15,0,t_] = ( 0.5 * ( (1. + z1ee) * taum + tau_se )**(-2.) * factor_ee1 * 1e-3 * taum
                              * ( 2. * Jee_sq * tau_se * taum * state_[0,9,t_] ) * sigma_sqrt_e )
        # should not be state[0,9,t-1] because also computing sigma t-1
        jacobian_[15,1,t_] = ( 0.5 * ( (1. + z1ei) * taum + tau_si )**(-2.) * factor_ei1 * 1e-3 * taum
                              * ( 2. * Jei_sq * tau_si * taum * state_[0,10,t_] ) * sigma_sqrt_e )
        jacobian_[15,9,t_] = - 0.5 * 2. * Jee_sq * tau_se * taum * ( (1. + z1ee) * taum + tau_se )**(-1.) * sigma_sqrt_e
        jacobian_[15,10,t_] = - 0.5 * 2. * Jei_sq * tau_si * taum * ( (1. + z1ei) * taum + tau_si )**(-1.) * sigma_sqrt_e
        
        sigma_ie = 2. * Jie_sq * state_[0,11,t_] * tau_se * taum / ((1. + z1ie) * taum + tau_se)
        sigma_ii = 2. * Jii_sq * state_[0,12,t_] * tau_si * taum * ( (1. + z1ii ) * taum + tau_si )**(-1)
        
        arg = ( sigma_ie + sigma_ii + sigmai_ext**2 )
        if arg > 0:
            sigma_sqrt_i = arg**(-1./2.)
        else:
            sigma_sqrt_i = 0.
            print("WARNING: sigma i smaller zero")
        
        jacobian_[16,0,t_] = ( 0.5 * ( (1. + z1ie) * taum + tau_se )**(-2.) * factor_ie1 * 1e-3 * taum
                              * ( 2. * Jie_sq * tau_se * taum * state_[0,11,t_] ) * sigma_sqrt_i )
        jacobian_[16,1,t_] = ( 0.5 * ( (1. + z1ii) * taum + tau_si )**(-2.) * factor_ii1 * 1e-3 * taum
                              * ( 2. * Jii_sq * tau_si * taum * state_[0,12,t_] ) * sigma_sqrt_i )
        jacobian_[16,11,t_] = - 0.5 * 2. * Jie_sq * tau_se * taum * ( (1. + z1ie) * taum + tau_se )**(-1.) * sigma_sqrt_i
        jacobian_[16,12,t_] = - 0.5 * 2. * Jii_sq * tau_si * taum * ( (1. + z1ii) * taum + tau_si )**(-1.) * sigma_sqrt_i
        
        jacobian_[17,2,t_] = - d_V_func_mu(state_[0,2,t_] - state_[0,4,t_]/C, sigmarange, ds,
                                          state_[0,15,t_], Irange, dI, C, precalc_V, interpolate_V)
        jacobian_[17,4,t_] = d_V_func_mu(state_[0,2,t_] - state_[0,4,t_]/C, sigmarange, ds,
                                          state_[0,15,t_], Irange, dI, C, precalc_V, interpolate_V) / C
        jacobian_[17,15,t_] = - d_V_func_sigma(state_[0,2,t_] - state_[0,4,t_]/C, sigmarange, ds,
                                          state_[0,15,t_], Irange, dI, C, precalc_V, interpolate_V)
        
        jacobian_[18,2,t_] = - d_tau_func_mu(state_[0,2,t_] - state_[0,4,t_]/C, sigmarange, ds,
                                          state_[0,15,t_], Irange, dI, C, precalc_tau_mu, interpolate_tau)
        jacobian_[18,4,t_] = d_tau_func_mu(state_[0,2,t_] - state_[0,4,t_]/C, sigmarange, ds,
                                          state_[0,15,t_], Irange, dI, C, precalc_tau_mu, interpolate_tau) / C
        jacobian_[18,15,t_] = - d_tau_func_sigma(state_[0,2,t_] - state_[0,4,t_]/C, sigmarange, ds,
                                          state_[0,15,t_], Irange, dI, C, precalc_tau_mu, interpolate_tau)
        
        jacobian_[19,3,t_] = - d_tau_func_mu(state_[0,3,t_], sigmarange, ds,
                                          state_[0,16,t_], Irange, dI, C, precalc_tau_mu, interpolate_tau)
        jacobian_[19,16,t_] = - d_tau_func_sigma(state_[0,3,t_], sigmarange, ds,
                                          state_[0,16,t_], Irange, dI, C, precalc_tau_mu, interpolate_tau)

    
    return jacobian_

@numba.njit
def d_r_func_mu(mu, sigmarange, ds, sigma, Irange, dI, C, precalc_r, interpolate_r):
    if interpolate_r:
        result = jac_aln.der_mu(sigma, sigmarange, ds, mu, Irange, dI, C, precalc_r)
        return result
    x_shift_mu = - 2.
    x_scale_mu = 0.6
    y_scale_mu = 0.1
    result = y_scale_mu * x_scale_mu / np.cosh(x_scale_mu * mu + x_shift_mu)**2
    #result = 1e-3
    return result

@numba.njit
def d_r_func_sigma(mu, sigmarange, ds, sigma, Irange, dI, C, precalc_r, interpolate_r):
    #return 1. * 1e-3
    #return (1. + 3. * sigma**2) * 1e-3
    if interpolate_r:
        result = jac_aln.der_sigma(sigma, sigmarange, ds, mu, Irange, dI, C, precalc_r)
        return result
    x_shift_sigma = -1.
    x_scale_sigma = 0.6
    y_scale_sigma = 1./2500.
    result = np.sinh(x_scale_sigma * sigma + x_shift_sigma) * y_scale_sigma * x_scale_sigma
    #result = 1e-3
    return result

@numba.njit
def d_tau_func_mu(mu, sigmarange, ds, sigma, Irange, dI, C, precalc_tau_mu, interpolate_tau):
    #return 1. - ( 1. + mu )**(-2.) #+ sigma
    if interpolate_tau:
        result = jac_aln.der_mu(sigma, sigmarange, ds, mu, Irange, dI, C, precalc_tau_mu)
        return result
    mu_shift = - 1.1
    sigma_scale = 0.5
    mu_scale = - 10
    mu_scale1 = - 3
    sigma_shift = 1.4
    result = sigma_scale * sigma + mu_scale1 + ( mu_scale / (sigma + sigma_shift) ) * np.exp( mu_scale * ( mu_shift + mu ) / ( sigma + sigma_shift ) )
    #result = 1e-3
    return result

@numba.njit
def d_tau_func_sigma(mu, sigmarange, ds, sigma, Irange, dI, C, precalc_tau_mu, interpolate_tau):
    #return 1. - ( 1. + sigma )**(-2.) #+ mu
    if interpolate_tau:
        result = jac_aln.der_sigma(sigma, sigmarange, ds, mu, Irange, dI, C, precalc_tau_mu)
        return result
    mu_shift = - 1.1
    sigma_scale = 0.5
    mu_scale = - 10
    sigma_shift = 1.4
    result = sigma_scale * ( mu_shift + mu ) - (mu_scale * (mu_shift + mu) / (sigma + sigma_shift)**2) * np.exp(
        mu_scale * ( mu_shift + mu ) / ( sigma + sigma_shift ) )  
    #result = 1e-3
    return result

@numba.njit
def d_V_func_mu(mu, sigmarange, ds, sigma, Irange, dI, C, precalc_V, interpolate_V):
    if interpolate_V:
        result = jac_aln.der_mu(sigma, sigmarange, ds, mu, Irange, dI, C, precalc_V)
        return result
    y_scale1 = 30.
    mu_shift1 = 1.
    y_scale2 = 2.
    mu_shift2 = 0.5
    sigma_shift = 0.1
    result = y_scale1 / np.cosh( mu + mu_shift1 )**2 - y_scale2 * 2. * ( mu - mu_shift2 ) * np.exp( - ( mu - mu_shift2 )**2 ) / ( sigma + sigma_shift )
    #result = 1e-3
    return result

@numba.njit
def d_V_func_sigma(mu, sigmarange, ds, sigma, Irange, dI, C, precalc_V, interpolate_V):
    if interpolate_V:
        result = jac_aln.der_sigma(sigma, sigmarange, ds, mu, Irange, dI, C, precalc_V)
        return result
    y_scale2 = 2.
    mu_shift2 = 0.5
    sigma_shift = 0.1
    result = - y_scale2 * np.exp( - ( mu - mu_shift2 )**2 ) / ( sigma + sigma_shift )**2
    #result = 1e-3
    return result