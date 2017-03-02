import matplotlib.pyplot as plt
from q1 import Neuron, voltage_time_graph
from math import exp

if __name__ == '__main__':
    n = Neuron()
    e = exp( 1 / n.tau_m )
    n.I_e = ((n.V_t - n.E_L)*e + n.E_L - n.V_r) / (n.R_m*e - n.R_m)
    print( 'The minimum value for I_e is %.1fe-9' % (n.I_e * 1e9) )
    voltage_time_graph( n, 1e-3, 1 )
