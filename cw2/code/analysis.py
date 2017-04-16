import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def load( name ):
    '''
    '''
    with open( '../data/%s.csv' % name ) as f:
        data = map( lambda x: float(x.strip()), f.readlines() )
    return np.array([ *data ])

def fire_rate( fire_times ):
    '''
    '''
    fire_times /= 1e4 # convert to seconds
    fire_times = np.int64( fire_times )
    return np.bincount( fire_times )

def plot_correlations():
    '''
    '''
    fig, axs = plt.subplots( 4, 4 )
    frs = [ fire_rate(load( 'neuron%d' % i ))
            for i in range( 1, 5 ) ]
    max_steps = max( fr.size for fr in frs )
    frs = [ np.concatenate(( fr, np.zeros((max_steps - fr.size)) ))
            for fr in frs ]
    kernel = np.ones(100) / 100
    for i, j in product( range(1, 5), range(1, 5) ):
        x = frs[ i - 1 ]
        y = frs[ j - 1 ]
        cor = np.correlate( x, y, mode = 'full' )
        cor_smooth = np.convolve( cor, kernel, mode = 'same' )
        cor = cor[ cor.size // 2 : ][ : 1200 ]
        cor_smooth  = cor_smooth[ cor_smooth.size // 2 : ][ : 1200 ]
        ax = axs[ i - 1, j - 1 ]
        ax.plot( cor, c='b', alpha=0.5 )
        ax.plot( cor_smooth, c='r' )
        ax.set_xlabel( 'Lag ($10^{-4} s$)' );
        if i == j:
            ax.set_title( 'Auto-correlation for Neuron %d' % i )
            ax.set_ylabel( 'Fire Rate Auto-correlation ($10^{-4} s^{-2}$)' )
        else:
            ax.set_title( 'Cross-correlation between Neurons %d and %d' % (i, j) )
            ax.set_ylabel( 'Fire Rate Cross-correlation ($10^{-4} s^{-2}$)' )
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

def show_sfrm( neuron ):
    '''
    '''
    plt.matshow(spatial_fire_rate_matrix(load(neuron)))
    plt.colorbar()
    plt.xlabel( 'X' )
    plt.ylabel( 'Y' )
    plt.title( 'Fire rate of %s with respect to position' % neuron )
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
    dt = t[ 1: ] - t[ :-1 ]
    v = dx / np.vstack(( dt, dt )).T
    return v

def speed():
    '''
    '''
    v = velocity()
    v = np.sqrt( v[:, 0]**2 + v[:, 1]**2 )
    v = np.concatenate((v, np.array([0])))
    return v
