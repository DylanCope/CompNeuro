import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import product

def load( name ):
    '''
    '''
    with open( '../data/%s.csv' % name ) as f:
        data = map( lambda x: float(x.strip()), f.readlines() )
    return np.array([ *data ])

def fire_rate( fire_times, interval = 1 ):
    '''
    '''
    fire_times -= load('time')[0]
    fire_times /= interval * 1e4 # convert to seconds
    fire_times = np.int64( fire_times )
    return np.bincount( fire_times )

def plot_spike_trains():
    fig, axs = plt.subplots( 2, 2 )
    frs = [ fire_rate(load( 'neuron%d' % i ))
            for i in range( 1, 5 ) ]

    for i, j in product( range(0, 2), range(0, 2) ):
        ax = axs[ i, j ]
        idx = i + 2*j
        ax.plot( frs[idx] )
        ax.set_xlabel( 'Time ($s$)' )
        ax.set_ylabel( 'Spikes ($s^{-1}$)')
        ax.set_title( 'Spike Train for %d' % (idx + 1) )
    plt.show()

def plot_correlations( time_interval = 1, num_seconds = 5 ):
    '''
    '''
    fig, axs = plt.subplots( 4, 4 )
    num_bins = int( num_seconds / time_interval )
    frs = [ fire_rate(load( 'neuron%d' % i ), time_interval )[ : 2*num_bins ]
            for i in range( 1, 5 ) ]

    # max_steps = max( fr.size for fr in frs )
    # frs = [ np.concatenate(( fr, np.zeros((max_steps - fr.size)) ))
    #         for fr in frs ]

    smooth_size = int(100*time_interval)
    kernel = np.ones( smooth_size ) / smooth_size
    n = num_bins
    for i, j in product( range(1, 5), range(1, 5) ):
        x = frs[ i - 1 ]
        y = frs[ j - 1 ]
        t = np.arange( num_bins ) * time_interval

        cor = np.correlate( x, y, mode = 'full' )
        cor = cor[ cor.size // 2 : ][ : num_bins ]
        ax = axs[ i - 1, j - 1 ]
        ax.plot( t, cor, c='b', alpha=0.5 )

        if smooth_size > 2:
            cor_smooth = np.convolve( cor, kernel, mode = 'same' )
            cor_smooth  = cor_smooth[ cor_smooth.size // 2 : ][ : num_bins ]
            ax.plot( t, cor_smooth, c='r' )

        ax.set_xlabel( 'Lag ($s$)' )
        if i == j:
            ax.set_title( 'Auto-correlation for Neuron %d' % i )
            ax.set_ylabel( 'Fire Rate Auto-correlation ($s^{-2}$)' )
        else:
            ax.set_title( 'Cross-correlation between Neurons %d and %d'
                          % (i, j) )
            ax.set_ylabel( 'Fire Rate Cross-correlation ($s^{-2}$)' )
    plt.show()

def fired_indices( time_stamps, fire_times ):
    '''
    '''
    # indices in x, y and t closest to when neuron fired
    find = np.vectorize( lambda v : np.argmin(np.abs( time_stamps - v )) )
    fired_idxs = find( fire_times )
    return fired_idxs

def firing_positions( fire_times ):
    '''
    '''
    time_stamps = load( 'time' )
    fired_idxs = fired_indices( time_stamps, fire_times )
    X = np.vstack(( load( 'x' )[ fired_idxs ],
                    load( 'y' )[ fired_idxs ] )).T
    return X

def spatial_fire_rate_matrix( fire_times ):
    '''
    '''
    fire_pos = firing_positions( fire_times )

    grid_shape = ( 300, 300 )
    rows, cols = np.indices( grid_shape )

    def manhattan( x ):
        return ( np.abs( cols - x[0] ) + np.abs( rows - x[1] ) ).flatten()

    dists = np.apply_along_axis( manhattan, 1, fire_pos )
    return np.sum( dists < 1, axis=0 ).reshape( grid_shape )

def plot_sfrms():
    '''
    '''
    fig, axs = plt.subplots( 2, 2 )
    for i, j in product( range(2), range(2) ):
        ax = axs[ i, j ]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        idx = i + 2*j
        n = load( 'neuron%d' % (idx + 1) )
        im = ax.matshow(spatial_fire_rate_matrix( n ))
        colorbar = plt.colorbar( im, cax = cax, orientation = 'vertical' )
        colorbar.set_label( 'Fire rate per unit area')
        ax.set_xlabel( 'X' )
        ax.set_ylabel( 'Y' )
        ax.set_title( 'Fire rate of neuron %s with respect to position' % (idx + 1) )
    plt.show()

def show_path( steps = -1 ):
    '''
    '''
    t = load( 'time' )[ : steps ]
    X = np.vstack(( load( 'x' )[ : steps ],
                    load( 'y' )[ : steps ],
                    ( t - t[0] ) / 1e4 )).T

    grid_shape = ( 300, 300 )
    grid = np.zeros( grid_shape )

    def trace( x ):
        grid[tuple(np.uint64(x[ : 2 ]))] = x[2]

    np.apply_along_axis( trace, 1, X )

    plt.matshow( grid )
    plt.colorbar()
    plt.xlabel( 'X' )
    plt.ylabel( 'Y' )
    plt.title( 'Time at position' )
    plt.show()

def velocity():
    '''
    '''
    X = np.vstack(( load( 'x' ),
                    load( 'y' ) )).T
    t = load( 'time' )

    dx = X[ 1: ] - X[ :-1 ]
    dt = (t[ 1: ] - t[ :-1 ]) / 1e4
    v = dx / np.vstack(( dt, dt )).T
    return v

def speed():
    '''
    '''
    v = velocity()
    v = np.sqrt( v[:, 0]**2 + v[:, 1]**2 )
    v = np.concatenate((v, np.array([0])))
    return v
