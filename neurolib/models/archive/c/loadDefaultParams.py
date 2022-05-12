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
    params.model = "c"
    params.name = "c"
    params.description = "Simplified version of simplified aln for testing purposes"
    
    params.N = 1
    params.Cmat = np.zeros((1, 1))
    params.lengthMat = np.zeros((1, 1))
    params.Dmat = np.zeros((1, 1))

    # runtime parameters
    params.dt = 0.1  # ms 0.1ms is reasonable
    params.duration = 2000  # Simulation duration (ms)

    # if zero, no handle on rates
    params.rates_exc_init = np.array( [[0.]] )
    params.mufe_init = np.array( [[1.]] )  # (linear) filtered mean input
    
    if lookupTableFileName is None:
        lookupTableFileName = os.path.join(os.path.dirname(__file__), "aln-precalc", "quantities_cascade.h5")

    hf = h5py.File(lookupTableFileName, "r")
    params.Irange = hf.get("mu_vals")[()]
    params.sigmarange = hf.get("sigma_vals")[()]
    
    factor = 1
    
    params.dI = factor * (params.Irange[1] - params.Irange[0] )
    params.ds = params.sigmarange[1] - params.sigmarange[0]
    
    params.Irange = np.arange(params.Irange[0],params.Irange[-1],params.dI)
    params.sigmarange = np.arange(params.sigmarange[0], params.sigmarange[-1], params.ds)
    
    params.precalc_r = hf.get("r_ss")[()][()]
    
    #params.precalc_r = np.zeros(( params.Irange.shape[0], params.sigmarange.shape[0] ))
    
    #for j in range(params.precalc_r.shape[1]):
        #params.precalc_r[:,j] = 0.001 * (3. + np.linspace(params.Irange[0],params.Irange[-1],params.precalc_r.shape[0]) )
        #params.precalc_r[:,j] = 0.001 * (3. + np.tanh(params.Irange))
    
    params.C = 200.0  # pF

    return params

