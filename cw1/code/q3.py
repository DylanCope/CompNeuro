import matplotlib.pyplot as plt
from q1 import Neuron, voltage_time_graph
from q2 import minimum_I_e

if __name__ == '__main__':
    fig, ax = plt.subplots()
    n = Neuron()
    n.I_e = minimum_I_e( n ) - 0.1e-9 #A
    voltage_time_graph( ax, n, 1e-3, 1 )
    plt.show()
