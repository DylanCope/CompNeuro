from q5 import ConnectedNeuron, plot_and_cosimulate, choose_rand_voltage
from math import exp
from random import seed
import matplotlib.pyplot as plt

class ConnectedAlphaNeuron( ConnectedNeuron ):

    def __init__( self ):
        super( ConnectedAlphaNeuron, self ).__init__()
        self.time_since_fire = 0
        self.not_fired_yet = True
        self.s_max = 0

    def fire( self ):
        ''' Removing the increment of P to s from firing
        '''
        super( ConnectedNeuron, self ).fire()
        self.time_since_fire = 0
        self.not_fired_yet = False

    def update( self, dt ):

        if self.not_fired_yet:
            self.s = 0
        else:
            t = self.time_since_fire
            self.s = t * exp( - t / self.tau_s )
            self.s_max = max( self.output_current(), self.s_max )
        self.time_since_fire += dt

        return super( ConnectedNeuron, self ).update( dt )

def connect_two_neurons():
    '''
        Initialise setup for two excitatory connected neurons with default parameters and random initial voltages
    '''
    n1 = ConnectedAlphaNeuron()
    n2 = ConnectedAlphaNeuron()
    n1.input_neuron = n2
    n2.input_neuron = n1
    # n1.V = choose_rand_voltage( n1 )
    # n2.V = choose_rand_voltage( n2 )
    n1.V = n1.V_r
    n2.V = n2.V_r + (n2.V_t - n2.V_r) / 2

    print( 'Neuron 1 initial membrane potential: %f' % n1.V )
    print( 'Neuron 2 initial membrane potential: %f' % n2.V )

    return n1, n2

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
    print( n1.s_max )

    # Case (b): Two inhibitory neurons feeding into each others inputs simulated over one second
    print( 'Case b:')
    seed( sd )
    n1, n2 = connect_two_neurons()
    n1.E_s = n2.E_s = -80e-3 # mV
    plot_and_cosimulate([ n1, n2 ])
    print( 'Neuron 1 spike count: %d' % n1.num_spikes )
    print( 'Neuron 2 spike count: %d' % n2.num_spikes )

    plt.show()
