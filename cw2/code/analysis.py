import numpy as np
import matplotlib.pyplot as plt

def load( name ):
    with open( '../data/%s.csv' % name ) as f:
        data = map( lambda x: float(x.strip()), f.readlines() )
    return np.array([ *data ])

def fire_rate( fire_times ):
    '''
    '''
    fire_times /= 1e4 # convert to seconds
    fire_times = np.int64( fire_times )
    return np.bincount( fire_times )

def fired_indices( time_stamps, fire_times ):
    '''
        Arguments:

        Returns:

    '''
    # indices in x, y and t closest to when neuron fired
    find = np.vectorize( lambda v : np.argmin(np.abs( time_stamps - v )) )
    fired_idxs = find( fire_times )
    return fired_idxs

def firing_positions( fire_times ):
    '''
        Arguments:

        Returns:

    '''

    time_stamps = load( 'time' )
    fired_idxs = fired_indices( time_stamps, fire_times )

    x = load( 'x' )
    y = load( 'y' )

    X = np.vstack(( x[ fired_idxs ], y[ fired_idxs ] )).T
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
    y = load( 'y' )[ : steps ]
    x = load( 'x' )[ : steps ]
    X = np.vstack(( x, y, (t - t[0]) / 1e4 )).T

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
