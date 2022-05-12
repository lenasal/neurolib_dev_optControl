#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pickle

path = os.getcwd().split(os.sep + 'GUI')[0]
if path not in sys.path:
    print("not here")
    sys.path.append(path)

from neurolib.models.aln import ALNModel
from neurolib.utils import plotFunctions as plotFunc
from neurolib.utils import costFunctions as cost
import neurolib.dashboard.functions as functions
import neurolib.dashboard.data as data


# In[2]:


grid_data_file = os.path.join(os.getcwd().split('data')[0], 'bi.pickle')
with open(grid_data_file,'rb') as f:
    load_array= pickle.load(f)
exc = [load_array[0][20], load_array[0][50]]
inh = [load_array[1][20], load_array[1][50]]

folder = str(os.getcwd().split('data_2')[1])
print(folder)
case = str(folder[1])

file_ = 'down.pickle'

c_var = [0,1]
if len(folder) == 4:
    if folder[3] == 'e':
        p_var = [0]
    elif folder[3] == 'i':
        p_var = [1]
else:
    p_var = [0,1]
    
print(p_var)

step1 = 100
step2 = 10
step3 = 1


# In[3]:


aln = ALNModel()
N = aln.params.N

data.set_parameters(aln)

state_vars = aln.state_vars
init_vars = aln.init_vars

##############################################################
def setinit(init_vars_, model):
    state_vars = model.state_vars
    init_vars = model.init_vars
    for iv in range(len(init_vars)):
        for sv in range(len(state_vars)):
            if state_vars[sv] in init_vars[iv]:
                #print("set init vars ", )
                if model.params[init_vars[iv]].ndim == 2:
                    model.params[init_vars[iv]][0,:] = init_vars_[sv]
                else:
                    model.params[init_vars[iv]][0] = init_vars_[sv]
                    
##############################################################               
def setmaxmincontrol(max_c_c, min_c_c, max_c_r, min_c_r):
    import numpy as np
    
    max_cntrl = np.zeros(( 6 ))
    min_cntrl = np.zeros(( 6 ))
    
    max_cntrl[0] = max_c_c
    min_cntrl[0] = min_c_c
    max_cntrl[1] = max_c_c
    min_cntrl[1] = min_c_c
    max_cntrl[2] = max_c_r
    min_cntrl[2] = min_c_r
    max_cntrl[3] = max_c_r
    min_cntrl[3] = min_c_r
    max_cntrl[4] = max_c_r
    min_cntrl[4] = min_c_r
    max_cntrl[5] = max_c_r
    min_cntrl[5] = min_c_r
            
    return max_cntrl, min_cntrl


# In[4]:


d_array = np.arange(500., 20, -0.1)
t0_array = d_array - 20.
t_pen = np.zeros(( t0_array.shape ))
for i in range(len(t_pen)):
    t_pen[i] = t0_array[i] / d_array[i]


# In[5]:


dur_pre = 10
dur_post = 10

n_pre = int(np.around(dur_pre/aln.params.dt + 1.,1))
n_post = int(np.around(dur_post/aln.params.dt + 1.,1))

tol = 1e-12
start_step = 10.
c_scheme = np.zeros(( 1,1 ))
c_scheme[0,0] = 1.
u_mat = np.identity(1)
u_scheme = np.array([[1.]])

if case in ['1', '2']:    # low to high
    max_I = [3., -3.]
else:
    max_I = [-3., 3.]
    
if case in ['1', '3']:    # sparsity
    factor_ws = 1.
    factor_we = 0.
else:
    factor_ws = 0.
    factor_we = 1.
    
maxC = [1000., -1000., 0.18, 0.]
max_cntrl, min_cntrl = setmaxmincontrol(maxC[0], maxC[1], maxC[2], maxC[3])


# In[6]:


bestControl_ = np.array( [[None] * len(t_pen)] * len(exc) )
bestState_ = np.array( [[None] * len(t_pen)] * len(exc) )
cost_ = np.array( [[None] * len(t_pen)] * len(exc) )
costnode_ = np.array( [[None] * len(t_pen)] * len(exc) )
weights_ = np.array( [[None] * len(t_pen)] * len(exc) )
convergence_ = np.array( [[None] * len(t_pen)] * len(exc) )

initVars = [None] * len(exc)
target = [None] * len(exc)
cost_uncontrolled = [None] * len(exc)

print(bestControl_.shape)


# In[7]:


if os.path.isfile(file_) :
    print("file found")
    
    with open(file_,'rb') as f:
        load_array = pickle.load(f)

    bestControl_ = load_array[0]
    bestState_ = load_array[1]
    cost_ = load_array[2]
    costnode_ = load_array[3]
    weights_ = load_array[4]


# In[ ]:


# get initial parameters and target states

i_range = range(len(exc))
data.set_parameters(aln)
max_percent = []

for i in i_range:
    print("------- ", i, exc[i], inh[i])
    aln.params.ext_exc_current = exc[i] * 5.
    aln.params.ext_inh_current = inh[i] * 5.
    
    aln.params.duration = 3000.
    
    control0 = aln.getZeroControl()
    control0 = functions.step_control(aln, maxI_ = max_I[0])

    aln.run(control=control0)
    
    target_rates = np.zeros((2))
    target_rates[0] = aln.rates_exc[0,-1] 
    target_rates[1] = aln.rates_inh[0,-1]

    control0 = functions.step_control(aln, maxI_ = max_I[1])
    aln.run(control=control0)

    init_state_vars = np.zeros(( len(state_vars) ))
    for j in range(len(state_vars)):
        if aln.state[state_vars[j]].size == 1:
            init_state_vars[j] = aln.state[state_vars[j]][0]
        else:
            init_state_vars[j] = aln.state[state_vars[j]][0,-1]

    initVars[i] = init_state_vars
    target[i] = [target_rates[0], target_rates[1]]
    
    max_percent.append( (np.argmax(np.abs(bestControl_[i][0]), axis=2)[0,1] -100) / 4800 )


# In[ ]:


# get uncontrolled cost

data.set_parameters(aln)

for i in i_range:
    print("------- ", i, exc[i], inh[i])
    aln.params.ext_exc_current = exc[i] * 5.
    aln.params.ext_inh_current = inh[i] * 5.
    
    dur = d_array[0]
    aln.params.duration = dur
    
    target_ = aln.getZeroTarget()
    target_[:,0,:] = target[i][0]
    target_[:,1,:] = target[i][1]
            
    cost.setParams(1.0, 0.0, 0.0)

    setinit(initVars[i], aln)
    control0 = aln.getZeroControl()

    # "HS", "FR", "PR", "HZ"
    cgv = None
    max_it = 0

    bestControl_init_, bestState_init_, cost_init_, runtime_init_, grad_init_, phi_init_, phi1_, costnode_init_ = aln.A1(
        control0, target_, c_scheme, u_mat, u_scheme, max_iteration_ = max_it, tolerance_ = tol,
        startStep_ = start_step, max_control_ = max_cntrl, min_control_ = min_cntrl, t_sim_ = dur,
        t_sim_pre_ = dur_pre, t_sim_post_ = dur_post, CGVar = cgv, control_variables_ = c_var,
        prec_variables_ = p_var, transition_time_ = t_pen[0])
    
    cost_uncontrolled[i] = cost_init_[0]


# In[ ]:


for i in i_range:
    print("------- ", i, exc[i], inh[i])
    aln.params.ext_exc_current = exc[i] * 5.
    aln.params.ext_inh_current = inh[i] * 5.
    
    j = 0
    step = step1

    while j < 4799:

        if j == 0:
            j += step
            continue
            
        if not convergence_[i][j-step]:
            if j - step > 0:
                j -= step
                print('go back to -----', j)

        print('-----', j)

        if convergence_[i][j]:
            j += step
            continue

        dur = d_array[j]
        T = 5001 - j
        aln.params.duration = dur

        cost.setParams(1., 0., 1.)
        weights_[i][j] = cost.getParams()

        setinit(initVars[i], aln)

        max_it = 10000

        target_ = aln.getZeroTarget()
        target_[:,0,:] = target[i][0]
        target_[:,1,:] = target[i][1]

        if type(bestControl_[i][j]) == type(None):
            max_percent_prev = (np.argmax(np.abs(bestControl_[i][j-step]), axis=2)[0,1] - 100) / (T-200)
            shift_ind = 0
            if j < 3000:
                while max_percent_prev > max_percent[i]:
                    shift_ind += 1
                    max_percent_prev = (np.argmax(np.abs(bestControl_[i][j-step]), axis=2)[0,1] - 100 - shift_ind) / (T-200)

            control0 = np.zeros(( bestControl_[i][j-step][:,:,100:-(100+step)].shape ))
            print('shift ind = ', shift_ind)
            if shift_ind != 0:
                control0[:,:,:-shift_ind] = 1. * bestControl_[i][j-step][:,:,100+shift_ind:-(100+step)]
            else:
                control0[:,:,:] = 1. * bestControl_[i][j-step][:,:,100:-(100+step)]
                #control0[0,0,:] = 0.
        else:
            if bestControl_[i][j].shape[2] == T+200:
                control0 = bestControl_[i][j][:,:,100:-100]
                #control0[0,1,:] = 0.
            else:
                control0 = bestControl_[i][j][:,:,100:-(100+j)]

        bestControl_[i][j], bestState_[i][j], cost_[i][j], runtime_, grad_, phi_, phi1_, costnode_[i][j] = aln.A1(
            control0, target_, c_scheme, u_mat, u_scheme, max_iteration_ = max_it, tolerance_ = tol,
            startStep_ = start_step, max_control_ = max_cntrl, min_control_ = min_cntrl, t_sim_ = dur,
            t_sim_pre_ = dur_pre, t_sim_post_ = dur_post, CGVar = cgv, control_variables_ = c_var,
            prec_variables_ = p_var, transition_time_ = t_pen[j])
        
        time_ = np.arange(-dur_pre,round(dur+dur_post+aln.params.dt,1),aln.params.dt)
        target_ = aln.getZeroTarget()
        target_[:,0,:] = target[i][0]
        target_[:,1,:] = target[i][1]

        print(costnode_[i][j][0][0][0], costnode_[i][j][2][0][:2])

        if cost_[i][j][-1] == 0.:
            convergence_[i][j] = True
            if j >= 4400:
                step = step2
            if j >=4790:
                step = step3

        with open(file_,'wb') as f:
            pickle.dump([bestControl_, bestState_, cost_, costnode_, weights_], f)

        j += step


# In[ ]:


bestControl_maxW = np.array( [[None] * len(t_pen)] * len(exc) )
bestState_maxW = np.array( [[None] * len(t_pen)] * len(exc) )
cost_maxW = np.array( [[None] * len(t_pen)] * len(exc) )
costnode_maxW = np.array( [[None] * len(t_pen)] * len(exc) )
weights_maxW = np.array( [[None] * len(t_pen)] * len(exc) )
convergence_maxW = np.array( [[None] * len(t_pen)] * len(exc) )

file_maxW = 'down_Wmax.pickle'


# In[ ]:


if os.path.isfile(file_maxW) :
    print("file found")
    
    with open(file_maxW,'rb') as f:
        load_array = pickle.load(f)

    bestControl_maxW = load_array[0]
    bestState_maxW = load_array[1]
    cost_maxW = load_array[2]
    costnode_maxW = load_array[3]
    weights_maxW = load_array[4]


# In[ ]:


for i in i_range:
    print('------------', i)
    j = 0

    aln.params.ext_exc_current = exc[i] * 5.
    aln.params.ext_inh_current = inh[i] * 5.

    aln.params.duration = dur

    while not convergence_maxW[i][j]:
        dur = d_array[j]
        aln.params.duration = dur

        setinit(initVars[i], aln)

        target_ = aln.getZeroTarget()
        target_[:,0,:] = target[i][0]
        target_[:,1,:] = target[i][1]

        control0 = aln.getZeroControl()
        if type(bestControl_maxW[i][j]) == type(None):
            max_it = 10
            control0 = bestControl_[i][0][:,:,n_pre-1:-n_post+1]
            weight_ = ( cost_uncontrolled[i] * (1. - t_pen[0]) / (1. - t_pen[j]) 
                       - sum(costnode_[i][j][0][0][:]) ) / sum( costnode_[i][j][2][0][:] )
            cost.setParams(1., weight_ * factor_we, weight_ * factor_ws)
            weights_maxW[i][j] = cost.getParams()
        else:
            control0 = bestControl_maxW[i][j][:,:,n_pre-1:-n_post+1]
            weight_ = ( cost_uncontrolled[i] * (1. - t_pen[0]) / (1. - t_pen[j])
                       - sum(costnode_maxW[i][j][0][0][:]) ) / sum( costnode_maxW[i][j][2][0][:] )
            cost.setParams(1., weight_ * factor_we, weight_ * factor_ws)
            weights_maxW[i][j] = cost.getParams()
            max_it = int(100)

        bestControl_maxW[i][j], bestState_maxW[i][j], cost_maxW[i][j], runtime_, grad_, phi_, phi1_, costnode_maxW[i][j] = aln.A1(
            control0, target_, c_scheme, u_mat, u_scheme, max_iteration_ = max_it, tolerance_ = tol,
            startStep_ = start_step, max_control_ = max_cntrl, min_control_ = min_cntrl, t_sim_ = dur,
            t_sim_pre_ = dur_pre, t_sim_post_ = dur_post, CGVar = cgv, control_variables_ = c_var,
            prec_variables_ = p_var, transition_time_ = t_pen[j])

        if cost_maxW[i][j][2] == 0.:
            convergence_maxW[i][j] = True

        print('cost splitting : ', costnode_maxW[i][j][0][0][0], costnode_maxW[i][j][2][0][:2])

        with open(file_maxW,'wb') as f:
            pickle.dump([bestControl_maxW, bestState_maxW, cost_maxW, costnode_maxW, weights_maxW], f)


# In[ ]:


for i in i_range:
    print("------- ", i, exc[i], inh[i])
    aln.params.ext_exc_current = exc[i] * 5.
    aln.params.ext_inh_current = inh[i] * 5.
    
    j = 0
    step = step1

    while j < 4800:

        if j == 0:
            j += step
            continue
            
        if not convergence_maxW[i][j-step]:
            if j - step > 0:
                j -= step
                print('go back to -----', j)

        print('-----', j)

        if convergence_maxW[i][j]:
            j += step
            continue

        dur = d_array[j]
        T = 5001 - j
        aln.params.duration = dur
        setinit(initVars[i], aln)

        max_it = 1000

        target_ = aln.getZeroTarget()
        target_[:,0,:] = target[i][0]
        target_[:,1,:] = target[i][1]

        if type(bestControl_maxW[i][j]) == type(None):
            max_percent_prev = (np.argmax(np.abs(bestControl_maxW[i][j-step]), axis=2)[0,1] - 100) / (T-200)
            shift_ind = 0
            if j < 4000:
                while max_percent_prev > max_percent[i]:
                    shift_ind += 1
                    max_percent_prev = (np.argmax(np.abs(bestControl_maxW[i][j-step]), axis=2)[0,1] - 100 - shift_ind) / (T-200)

            control0 = np.zeros(( bestControl_maxW[i][j-step][:,:,100:-(100+step)].shape ))
            print('shift ind = ', shift_ind)
            if shift_ind != 0:
                control0[:,:,:-shift_ind] = 1. * bestControl_maxW[i][j-step][:,:,100+shift_ind:-(100+step)]
            else:
                control0[:,:,:] = 1. * bestControl_maxW[i][j-step][:,:,100:-(100+step)]
                
            weight_ = ( cost_uncontrolled[i] * (1. - t_pen[0]) / (1. - t_pen[j])
                          - sum(costnode_maxW[i][j-step][0][0][:]) ) / sum( costnode_maxW[i][j-step][2][0][:] )
            cost.setParams(1., weight_ * factor_we, weight_ * factor_ws)
            weights_maxW[i][j] = cost.getParams()
        
        else:
            if bestControl_maxW[i][j].shape[2] == T+200:
                control0 = bestControl_maxW[i][j][:,:,100:-100]
                #control0[0,1,:] = 0.
            else:
                control0 = bestControl_maxW[i][j][:,:,100:-(100+j)]
                
                weight_ = ( cost_uncontrolled[i] * (1. - t_pen[0]) / (1. - t_pen[j])
                           - sum(costnode_maxW[i][j][0][0][:]) ) / sum( costnode_maxW[i][j][2][0][:] )
                cost.setParams(1., weight_ * factor_we, weight_ * factor_ws)
                weights_maxW[i][j] = cost.getParams()

        bestControl_maxW[i][j], bestState_maxW[i][j], cost_maxW[i][j], runtime_, grad_, phi_, phi1_, costnode_maxW[i][j] = aln.A1(
            control0, target_, c_scheme, u_mat, u_scheme, max_iteration_ = max_it, tolerance_ = tol,
            startStep_ = start_step, max_control_ = max_cntrl, min_control_ = min_cntrl, t_sim_ = dur,
            t_sim_pre_ = dur_pre, t_sim_post_ = dur_post, CGVar = cgv, control_variables_ = c_var,
            prec_variables_ = p_var, transition_time_ = t_pen[j])

        print(costnode_maxW[i][j][0][0][:2], costnode_maxW[i][j][2][0][:2])

        if cost_maxW[i][j][2] == 0.:
            convergence_maxW[i][j] = True
            if j >= 4400:
                step = step2
            if j >=4790:
                step = step3

        with open(file_maxW,'wb') as f:
            pickle.dump([bestControl_maxW, bestState_maxW, cost_maxW, costnode_maxW, weights_maxW], f)

        j += step


# In[ ]:




