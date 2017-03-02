import numpy as np
import matplotlib.pyplot as plt
from operator import iadd

if __name__ == '__main__':

    # neuron parameters
    tau_m = 10e-3 # s (membrane time constant)
    E_L = V_r = -70e-3 # V (leak potential and rest voltage)
    V_t = -40e-3 # V (spiking threshold)
    R_m = 10e6 # Ohms (membrane resistance)
    I_e = 3.1e-9 # A (injected current ('e' for electrode))

    # Using the integrate and fire model this code plots voltage in the cell over time, no firing behaivour or refractory rest period is simulated for simplicity. Once the membrane potential exceeds threshold it's value is set to V_r

    dt = 1e-3 # s (time step for Euler method)
    T = 1 # s (finish time simulation)

    v = V_r

    def V( t ):
        global v
        dvdt = (E_L - v + R_m*I_e) / tau_m
        v = v + dvdt * dt if v < V_t else V_r
        return v

    ts = [ i * dt for i in range( 1 + int( T / dt ) ) ]
    vs = [ V(t) for t in ts ]

    fig, ax = plt.subplots()
    ax.set_ylabel( 'Voltage (V)' )
    ax.set_xlabel( 'Time (s)' )
    ax.plot( ts, vs )

    plt.show()
