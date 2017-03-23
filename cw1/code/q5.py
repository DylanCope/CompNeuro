from q1 import Neuron as SimpleNeuron
import matplotlib.pyplot as plt
from random import seed, random
from numpy import arange

class ConnectedNeuron( SimpleNeuron ):
    '''
        A class to hold the properties of a neuron being simulated under the input current of the.
    '''
    def __init__( self, other ):
        super( ConnectedNeuron, self ).__init__()

        self.input_neuron = other

        # membrane parameters
        self.tau_m = 20e-3 # ms (membrane time constant)
        self.E_L = -70e-3 # mV (leak potential)
        self.V_r = -80e-3 # mV (rest voltage)
        self.V_t = -54e-3 # mV (spiking threshold)
        self.RI = 18e-3 # mV (the product of the membrane resistance R_m and the injected curent I_e)

        # synaptic parameters
        self.Rg = 0.15 # (the product of the membrane resistance R_m and a constant g_s which describes the 'strength' of a synapse)
        self.P = 0.5
        self.tau_s = 10e-3 #ms
        self.E_s = 0 # V (reverse potential) if 0 then the neuron is excitatory, if -80e-3 it is inhibitory

        # synaptic variables
        self.s = 0 # (post-synaptic conductance)

    def fire( self ):
        ''' Overrides the super fire method to include incrementing the post-synaptic conductance by a constant value.
        '''
        super( ConnectedNeuron, self ).fire()
        self.s += self.P

    def alpha_function( self ):
        ''' Returns the post-synaptic conductivity given the current state of the neuron.
        '''
        g_s = self.Rg / self.R_m
        return g_s * self.s * (self.E_s - self.V)

    def update( self, dt ):
        ''' Given the change in time since the last call to this method, the internal state of the neuron and returns the current internal voltage.
        '''
        # Using Euler's method to compute the updated post-synaptic conductance
        dsdt = - self.s / self.tau_s
        self.s = self.s + dsdt * dt

        I = self.RI / self.R_m
        self.I_e = self.input_neuron.alpha_function() + I
        return super( ConnectedNeuron, self ).update( dt )

def choose_rand_voltage( n ):
    '''
        Given a neuron n, a random voltage value is returned in the range n.V_r and n.V_t
    '''
    return n.V_r + (n.V_t - n.V_r) * random()

def connect_two_neurons():
    '''
        Initialise setup for two excitatory connected neurons with default parameters and random initial voltages
    '''
    n1 = ConnectedNeuron( None )
    n2 = ConnectedNeuron( n1 )
    n1.input_neuron = n2
    # n1.V = choose_rand_voltage( n1 )
    # n2.V = choose_rand_voltage( n2 )
    n1.V = n1.V_r
    n2.V = n2.V_r + (n2.V_t - n2.V_r) / 2

    print( 'Neuron 1 initial membrane potential: %f' % n1.V )
    print( 'Neuron 2 initial membrane potential: %f' % n2.V )

    return n1, n2

def cosimulate_neurons( neurons, dt, T ):
    ''' simulates a list of neurons over a given time, on each timestep all the neurons are updated
    Returns:
        ts: an ordered list of all the time values at each timestep
        vss: a list of lists, where each vss[i] is the list of voltages associated with each given neuron at the corresponding ts[i] iteration
        neurons: the list of updated neurons
    '''
    ts = arange( 0, T, dt )
    vss = [ [ n.update(dt) for n in neurons ] for t in ts ]
    return ts, vss, neurons

def plot_and_cosimulate( neurons ):

    ts, vss, neurons = cosimulate_neurons( neurons, 1e-3, 0.2 )
    vs1 = [ v1 for v1, v2 in vss ]
    vs2 = [ v2 for v1, v2 in vss ]

    fig, ax = plt.subplots( figsize = (10, 10) )
    ax.plot( ts, vs1, label = 'N1 Membrane Potential' )
    ax.plot( ts, vs2, label = 'N2 Membrane Potential' )
    ax.set_xlabel( 'Time (s)' )
    ax.set_ylabel( 'Voltage (V)' )
    ax.legend()

if __name__ == '__main__':

    sd = 10

    # Case (a): Two excitatory neurons feeding into each others inputs simulated over one second
    print( 'Case a:')
    seed( sd )
    n1, n2 = connect_two_neurons()
    plot_and_cosimulate([ n1, n2 ])
    print( 'Neuron 1 spike count: %d' % n1.num_spikes )
    print( 'Neuron 2 spike count: %d' % n2.num_spikes )
    print

    # Case (b): Two inhibitory neurons feeding into each others inputs simulated over one second
    print( 'Case b:')
    seed( sd )
    n1, n2 = connect_two_neurons()
    n1.E_s = n2.E_s = 80e-3 # mV
    plot_and_cosimulate([ n1, n2 ])
    print( 'Neuron 1 spike count: %d' % n1.num_spikes )
    print( 'Neuron 2 spike count: %d' % n2.num_spikes )

    plt.show()
