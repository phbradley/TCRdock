import numpy as np

# TCRdiv is a TCR repertoire diversity measure
#
def compute_tcrdiv(D, sigma=120.):
    ''' D is a symmetric matrix of TCRdist distances
    sigma is the width of the Gaussian smoothing term
    '''
    N = D.shape[0]
    D = D.copy() # don't change passed array
    D[np.arange(N), np.arange(N)] = 1e6 # set diagonal to a very large value
    return -1*np.log(np.sum(np.exp(-1*(D/sigma)**2))/(N*(N-1)))



# this is the algorithm used to select 50 representative TCRs from larger repertoires
#
def pick_reps(D, num_reps=50, sdev_big=120., sdev_small=36., min_size=0.5):
    ''' D is a symmetric distance matrix (e.g., of TCRdist distances)
    num_reps is the number of representatives to choose
    sdev_big defines the neighbor-density sum used for ranking
    sdev_small limits the redundancy
    both sdev_big and sdev_small are in distance units (ie same units as D)
    '''
    # the weight remaining for each instance
    wts = np.array([1.0]*D.shape[0])

    reps, sizes = [], []
    for ii in range(num_reps):
        if np.sum(wts)<1e-2:
            break
        gauss_big   = np.exp(-1*(D/sdev_big  )**2) * wts[:,None] * wts[None,:]
        gauss_small = np.exp(-1*(D/sdev_small)**2) * wts[:,None] * wts[None,:]
        nbr_sum = np.sum(gauss_big, axis=1)
        rep = np.argmax(nbr_sum)
        size = nbr_sum[rep]
        if size<min_size:
            break
        wts = np.maximum(0.0, wts - gauss_small[rep,:]/wts[rep])
        assert wts[rep] < 1e-3
        reps.append(rep)
        sizes.append(size)
    return reps, sizes
