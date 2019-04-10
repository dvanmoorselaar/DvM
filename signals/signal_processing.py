import numpy as np
from scipy import special as sp

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def legendre(N,X):
    matrixReturn = np.zeros((N+1,X.shape[0]))
    for i in enumerate(X):
        currValues = sp.lpmn(N,N,i[1])
        matrixReturn[:,i[0]] = np.array([j[N] for j in currValues[0]])
    return matrixReturn	

def laplacian_filter(X, x, y, z, leg_order = 10, smoothing = 1e-5):
    '''Computes surface Laplacian of EEG data
        This function, which is an implementation of algorithms described by 
        Perrin, Pernier, Bertrand, and Echallier (1989) PubMed #2464490 is modified after
        the LAPLACIAN_PERRINX function as implemented in Matlab's EEGLAB 
    
    Arguments:
        X {array} -- EEG data (epochs X electrodes X time)
        x {array} -- x coordinates of electrode positions
        y {array} -- y coordinates of electrode positions
        z {array} -- z coordinates of electrode positions
    
    Keyword Arguments:
        leg_order {int} -- order of Legendre polynomial (default: {10}) [12 for > 100 electrodes] 
        smoothing {float} -- G smoothing parameter (lambda) (default: {1e-5})
    
    Returns:
        surf_lap {array} -- the surface Laplacian (second spatial derivative)
    '''

    # switch EEG dimensions (electrodes must be first dimension to create laplacian)
    X = np.swapaxes(X, 0,1)

    num_elec = x.size

    # compute G and H matrices
    G = np.zeros((num_elec,num_elec))
    H = np.zeros((num_elec,num_elec))
    cosdist = np.zeros((num_elec,num_elec))

    # set default parameters for +/- 100 electrodes
    m = 3 if num_elec > 100 else 4

    # scale XYZ coordinates to unit sphere
    _,_,spherical_radii = cart2sph(x, y,z)
    maxrad = max(spherical_radii)
    x /= maxrad
    y /= maxrad
    z /= maxrad

    for i in range(num_elec):
        for j in range(i+1,num_elec):
            cosdist[i,j] = 1 - ( ((x[i] - x[j])**2 + (y[i] - y[j])**2  + (z[i] - z[j])**2)/2.0)
    cosdist += cosdist.T + np.identity(num_elec) 	

    # compute Legendre polynomial 
    legpoly = np.zeros((leg_order, num_elec, num_elec))
    for l in range(leg_order):
        for i in range(num_elec):
            temp = legendre(l+ 1,cosdist[i,:])
            legpoly[l,i] = temp[0]

    # precompute electrode-independent variables
    twoN1 = 2 * np.arange(1, leg_order + 1) + 1
    gdenom = ((np.arange(1, leg_order + 1)) * ((np.arange(1, leg_order + 1) + 1)))**m
    hdenom = ((np.arange(1, leg_order + 1)) * ((np.arange(1, leg_order + 1) + 1)))**(m-1)

    for i in range(num_elec):
        for j in range(num_elec):
            g, h = 0,0
            for l in range(leg_order):
                # compute G and H terms
                g += (twoN1[l]* legpoly[l,i,j]) / gdenom[l]
                h -= (twoN1[l]* legpoly[l,i,j]) / hdenom[l]

            G[i,j] = g/(4*np.pi)
            H[i,j] = -h/(4*np.pi)

    # mirror matrix
    G += G
    H += H

    # correct for diagonal-double
    G -= np.identity(num_elec) * G[0,0]/2.0
    H -= np.identity(num_elec) * H[0,0]/2.0

    # compute laplacian
    # reshape data to electrodes X time/trials
    orig_size = X.shape
    X = np.reshape(X, (orig_size[0],-1), order = 'F')
    print ('computing the laplacian for {} electrodes'.format(orig_size[0]))
    if orig_size[0] != num_elec:
        Warning('Number electrodes in data and number of specified coordinates do not match')

    # add smoothing constant to diagonal
    # change G so output is unadulterated
    Gs = G + np.identity(num_elec)*smoothing

    # compute C matrix
    GsinvS = np.sum(np.linalg.inv(Gs), axis = 0)
    XGs = np.matmul(X.T, np.linalg.inv(Gs))
    C = XGs - np.matrix(np.sum(XGs, axis = 1)/np.sum(GsinvS)).T*GsinvS

    # compute surface Laplacian (and reshape to original data size)
    surf_lap = np.reshape(np.array(C*H.T).T, orig_size, order = 'F')

    # switch back EEG dimensions 
    surf_lap = np.swapaxes(surf_lap, 0,1)

    return surf_lap