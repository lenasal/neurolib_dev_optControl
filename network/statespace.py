import matplotlib.pyplot as plt
import numpy as np
import os
import sys

path = os.getcwd().split(os.sep + 'network')[0]
if path not in sys.path:
    sys.path.append(path)

from neurolib.models.aln import ALNModel
from neurolib.utils import costFunctions as cost

def initmodel(mue, mui):
    aln = ALNModel(Cmat = cmat, Dmat = dmat)

    N = aln.params.N
    aln.params.sigma_ou = 0.
    aln.params.a = 0.
    aln.params.b = 0.

    aln.params.ext_exc_current = 0.
    aln.params.ext_inh_current = 0.

    aln.params.mue_ext_mean = mue * 5.
    aln.params.mui_ext_mean = mui * 5.
    
    aln.params.de = 0.
    aln.params.di = 0.
    
    return aln

cmat = 1. * np.array( [[0., 1.], [1., 0.]] )
dmat = 0. * np.array( [[1., 1.], [1., 1.]] )
aln = initmodel(0., 0.)
N = aln.params.N

#state space 

rows, cols = 10, 10
mue_ = np.arange(0., 1., 0.1)
mui_ = np.arange(0., 1., 0.1)
#fig = plt.figure(constrained_layout=True, figsize=(120, 40), sharey=True, sharex=True)
#subfigs = fig.subfigures(rows, cols, wspace = 0.001, hspace = 0.001)

for i in range(rows):
    for j in range(cols):
        print(i,j)
        aln = initmodel(mue_[i], mui_[j])
        aln.params.duration = 10000.
        control0 = aln.getZeroControl()
        control0[0,0,30000:32000] = -5.
        control0[0,0,70000:72000] = 5.
        aln.run( control = control0 )

        #print(aln.rates_exc[1,-20:])

        fig, ax = plt.subplots(2,4, figsize=(30, 10), sharey=True)
        mean_values = np.zeros(( 2, 4 ))
        
        #ax = subfigs[i,j].subplots(2,2, sharey=True, sharex=True)
        ax[0,0].plot(aln.t[28000:30000], aln.rates_exc[0,28000:30000], lw=1, c='red', label='Node 0 exc')
        mean_values[0,0] = np.mean(aln.rates_exc[0,28000:30000])
        ax[0,1].plot(aln.t[-2000:], aln.rates_exc[0,-2000:], lw=1, c='red', label='Node 0 exc')
        mean_values[0,1] = np.mean(aln.rates_exc[0,-2000:])
        ax[0,2].plot(aln.t[28000:30000], aln.rates_inh[0,28000:30000], lw=1, c='blue', label='Node 0 inh')
        mean_values[0,2] = np.mean(aln.rates_inh[0,28000:30000])
        ax[0,3].plot(aln.t[-2000:], aln.rates_inh[0,-2000:], lw=1, c='blue', label='Node 0 inh')
        mean_values[0,3] = np.mean(aln.rates_inh[0,-2000:])

        ax[1,0].plot(aln.t[28000:30000], aln.rates_exc[1,28000:30000], lw=1, c='red', label='Node 1 exc')
        mean_values[1,0] = np.mean(aln.rates_exc[1,28000:30000])
        ax[1,1].plot(aln.t[-2000:], aln.rates_exc[1,-2000:], lw=1, c='red', label='Node 1 exc')
        mean_values[1,1] = np.mean(aln.rates_exc[1,-2000:])
        ax[1,2].plot(aln.t[28000:30000], aln.rates_inh[1,28000:30000], lw=1, c='blue', label='Node 1 inh')
        mean_values[1,2] = np.mean(aln.rates_inh[1,28000:30000])
        ax[1,3].plot(aln.t[-2000:], aln.rates_inh[1,-2000:], lw=1, c='blue', label='Node 1 inh')
        mean_values[1,3] = np.mean(aln.rates_inh[1,-2000:])
 
        ax[0,0].set_ylabel("Rate [Hz]")
        ax[1,0].set_ylabel("Rate [Hz]")

        for k in range(4):
            ax[1,k].set_xlabel("t [ms]")
            for l in range(2):
                ax[l,k].text(0.5,0.5, '{:.4f}'.format(mean_values[l,k]), transform=ax[l,k].transAxes)     

        title_ = r'$\mu_E = $' + '{:.2f}'.format(mue_[i]) + r'$, \mu_I = $' + '{:.2f}'.format(mui_[j])
        plt.suptitle(title_)
        file_ = '{:.2f}'.format(mue_[i]) + '_{:.2f}'.format(mui_[j]) + '.png'
        plt.savefig(file_, dpi=100)
        
        #subfigs[i,j].suptitle(r'$\mu_E = $' + str(mue_[i]) + r'$, \mu_I = $' + str(mui_[j]))

#plt.suptitle('State Space 2-node network')
#plt.savefig('statespace_2node.png', dpi=300)
