import matplotlib.pyplot as plt
from numpy import arange
from q1 import Neuron, simulate_neuron
from functools import partial

def measure_fire_rate( neuron, dt, T, I_e ):
    neuron.I_e = I_e
    _, _, neuron = simulate_neuron( neuron, dt, T )
    fr = neuron.num_spikes / T
    neuron.reset()
    return fr

if __name__ == '__main__':
    neuron = Neuron()
    inputs = arange( 2e-9, 5e-9, 0.1e-9 )
    mapfunc = partial( measure_fire_rate, neuron, 1e-3, 1 )
    fire_rates = list(map( mapfunc, inputs ))

    fig, ax = plt.subplots()
    ax.plot( inputs, fire_rates )
    ax.set_xlabel( 'Input Current (A)' )
    ax.set_ylabel( 'Fire rate ($s^{-1}$)' )
    plt.show()
