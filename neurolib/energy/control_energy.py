import numpy as np

#from . import functions as functions
#from neurolib.utils import costFunctions as cost

def current_metabolic_node_channel(model, control_, n_neurons_, node, channel):
    dt = model.params.dt
    factor = 1e-9 * model.params.C/1000. # in Ampere
    electric_charge = 1.602176634 * 1e-19 # Coulomb

    int = 0.

    for t in range(control_.shape[2]):
        int += np.abs(control_[node, channel, t]) * dt * 1e-3   # milli seconds

    int *= n_neurons_ * factor / electric_charge

    max_per_sec = max(np.abs(control_[node, channel, :])) * n_neurons_ * factor / electric_charge

    return int, max_per_sec


def rate_spike_generation(model, control_, n_neurons_, node, channel):
    dt = model.params.dt
    energy_per_spike_atp = 7.1 * 1e8 # atp molecules: "Energy Budget for grey matter", Attwell, Laughlin

    n_spikes = 0.
    for t in range(control_.shape[2]):
        n_spikes += np.abs(control_[node, channel, t]) * dt  # ms * kHz

    print(n_spikes)

    energy_atp = n_spikes * n_neurons_ * energy_per_spike_atp
    max_per_sec = max(np.abs(control_[node, channel, :])) * 1e3 * n_neurons_ * energy_per_spike_atp

    return energy_atp, max_per_sec