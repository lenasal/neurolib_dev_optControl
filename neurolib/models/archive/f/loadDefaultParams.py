import os
import numpy as np
import h5py

from ...utils.collections import dotdict


def loadDefaultParams(Cmat=None, Dmat=None, lookupTableFileName=None):
    """Load default parameters for a network of aLN nodes.
    :param Cmat: Structural connectivity matrix (adjacency matrix) of coupling strengths, will be normalized to 1. If not given, then a single node simulation will be assumed, defaults to None
    :type Cmat: numpy.ndarray, optional
    :param Dmat: Fiber length matrix, will be used for computing the delay matrix together with the signal transmission speed parameter `signalV`, defaults to None
    :type Dmat: numpy.ndarray, optional
    :param lookUpTableFileName: Filename of lookup table with aln non-linear transfer functions and other precomputed quantities., defaults to aln-precalc/quantities_cascade.h
    :type lookUpTableFileName: str, optional
    :param seed: Seed for the random number generator, defaults to None
    :type seed: int, optional
    
    :return: A dictionary with the default parameters of the model
    :rtype: dict
    """

    params = dotdict({})

    # Todo: Model metadata
    # recently added for easier simulation of aln and brian in pypet
    params.model = "f"
    params.name = "f"
    params.description = "Simplified version of simplified aln for testing purposes"
    
    params.N = 1
    params.Cmat = np.zeros((1, 1))
    params.lengthMat = np.zeros((1, 1))
    params.Dmat = np.zeros((1, 1))

    # runtime parameters
    params.dt = 0.1  # ms 0.1ms is reasonable
    params.duration = 2000  # Simulation duration (ms)
    
    params.delay = 1

    # if zero, no handle on rates
    params.rates_exc_init = np.array( [[0.]] )
    params.mufe_init = np.array( [[1.]] )  # (linear) filtered mean input
    params.seem_init = np.array( [[0.1]] )
    params.seev_init = np.array( [[0.0]] )
    params.ext_exc_current = 0.0
    
    params.sigmae_ext = 1.5
    
    if lookupTableFileName is None:
        lookupTableFileName = os.path.join(os.path.dirname(__file__), "aln-precalc", "quantities_cascade.h5")

    hf = h5py.File(lookupTableFileName, "r")
    params.Irange = hf.get("mu_vals")[()]
    params.sigmarange = hf.get("sigma_vals")[()]
    params.dI = params.Irange[1] - params.Irange[0]
    params.ds = params.sigmarange[1] - params.sigmarange[0]
    
    params.precalc_r = hf.get("r_ss")[()][()]
    params.precalc_V = hf.get("V_mean_ss")[()]
    params.precalc_tau_mu = hf.get("tau_mu_exp")[()]
    
    #params.precalc_r = np.ones(( params.precalc_r.shape ))
    #for j in range(params.precalc_r.shape[1]):
    #    params.precalc_r[:,j] = 0.02+0.01*np.linspace(params.Irange[0],params.Irange[-1],params.precalc_r.shape[0])
    
    
    # recurrent coupling parameters
    params.Ke = 800.0  # Number of excitatory inputs per neuron
    
    # synaptic time constants
    params.tau_se = 2.0  # ms  "EE = IE", for fixed delays
    
    # PSC amplitudes
    params.cee = 0.3  # mV/ms
    
    # Coupling strengths used in Cakan2020
    params.Jee_max = 2.43  # mV/ms
    
    params.C = 200.0  # pF
    params.gL = 10.0  # nS

    return params

