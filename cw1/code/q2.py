import matplotlib.pyplot as plt
from q1 import Neuron, voltage_time_graph
from math import exp

def minimum_I_e( n ):
    e = exp( 1 / n.tau_m )
    return ((n.V_t - n.E_L)*e + n.E_L - n.V_r) / (n.R_m*e - n.R_m)

if __name__ == '__main__':
    n = Neuron()
    n.I_e = minimum_I_e( n )
    print( 'The minimum value for I_e is %.1fe-9' % (n.I_e * 1e9) )
    fig, ax = plt.subplots()
    voltage_time_graph( ax, n, 1e-3, 1 )
    plt.show()
