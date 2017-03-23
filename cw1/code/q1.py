import matplotlib.pyplot as plt
from numpy import arange

class Neuron:
    '''
        A class to hold the properties of a neuron being simulated under a constant injected current.
    '''

    def __init__( self ):

        # constant parameters
        self.tau_m = 10e-3 # s (membrane time constant)
        self.E_L = self.V_r = -70e-3 # V (leak potential and rest voltage)
        self.V_t = -40e-3 # V (spiking threshold)
        self.R_m = 10e6 # Ohms (membrane resistance)
        self.I_e = 3.1e-9 # A (injected current ('e' for electrode))

        # neuron variables
        self.V = self.V_r # V (internal voltage within the neuron)
        self.num_spikes = 0

    def fire( self ):
        self.V = self.V_r
        self.num_spikes += 1

    def update( self, dt ):
        ''' updates and returns the neuron's internal voltage value, the updated value represents the voltage at the next time step
        '''

        dvdt = (self.E_L - self.V + self.R_m * self.I_e) / self.tau_m
        self.V = self.V + dvdt * dt

        if self.V >= self.V_t:
            self.fire()

        return self.V

    def reset( self ):
        ''' sets the internal voltage to the preset rest voltage '''
        self.V = self.V_r
        self.num_spikes = 0

def simulate_neuron( neuron, dt, T ):
    ''' simulates a neuron over a given time
    '''
    ts = arange( 0, T, dt )
    vs = [ neuron.update(dt) for t in ts ]
    return ts, vs, neuron

def voltage_time_graph( ax, neuron, dt, T ):
    ts, vs, neuron = simulate_neuron( neuron, dt, T )
    ax.set_ylabel( 'Voltage (V)' )
    ax.set_xlabel( 'Time (s)' )
    ax.set_ylim([ neuron.V_r - 0.01, neuron.V_t + 0.01 ])
    ax.plot( ts, vs )

if __name__ == '__main__':

    # Using the integrate and fire model this code plots voltage in the cell over time, no firing behaivour or refractory rest period is simulated for simplicity. Once the membrane potential exceeds threshold it's value is set to V_r

    neuron = Neuron()

    dt = 1e-3 # s (time step for Euler method)
    T = 1 # s (finish time simulation)

    fig, ax = plt.subplots()
    voltage_time_graph( ax, neuron, dt, T )
    print( 'number of spikes = %d' % neuron.num_spikes )
    plt.show()
