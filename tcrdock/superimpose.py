######################################################################################88

import os.path
from os import system
from os.path import exists
from pathlib import Path
import pandas as pd
import numpy as np

from . import geom_util
from numpy.linalg import norm

def amir_calc_U(xyz1, xyz2):
    """
    Calculate the rotation matrix that is used for superimposing two chains during rmsd computation.

    xyz2 is moved onto xyz1 (I think)

    """
    n_atom = xyz1.shape[0]
    assert xyz1.shape == xyz2.shape == (n_atom,3)

    # center to CA centroid
    xyz1 = xyz1 - xyz1.mean(0)
    xyz2 = xyz2 - xyz2.mean(0)

    # Computation of the covariance matrix
    C = xyz2.T @ xyz1

    # Compute optimal rotation matrix using SVD
    V, S, W = np.linalg.svd(C)

    # get sign to ensure right-handedness
    d = np.ones([3,3])
    d[:,-1] = np.sign(np.linalg.det(V)*np.linalg.det(W))

    # Rotation matrix U
    U = (d*V) @ W

    return U



def superimposition_transform(fixcoords, movcoords):
    '''
    returns rotation, translation that would orient movcoords_in onto fixcoords_in

    as in x--->rotation*x + translation

    '''
    assert fixcoords.shape == movcoords.shape

    R = amir_calc_U(fixcoords, movcoords).transpose()

    fix_com = np.mean(fixcoords, axis=0)
    mov_com = np.mean(movcoords, axis=0)

    v = fix_com - R@mov_com

    return R, v

def superimpose_coords(fixcoords, movcoords):
    ''' Move movcoords onto fixcoords

    return copy of movcoords after superimposing

    '''

    N = len(fixcoords)
    assert N == len(movcoords)

    # this should not change fixcoords/movcoords
    R, translation = superimposition_transform(fixcoords, movcoords)

    return (R@movcoords.T).T + translation[None,:]



def get_symmetry_stub_from_coords(
        coords, # numpy array
        verbose=False,
        point_towards=None,
):
    ''' returns a stub aka
    dict {'axes':numpy array with axes as ROW vectors,
          'origin': vector}
    '''

    nalign = len(coords)//2
    assert coords.shape == (2*nalign, 3)

    acom = np.mean(coords[:nalign], axis=0)
    bcom = np.mean(coords[nalign:], axis=0)

    coords1 = coords
    coords2 = np.vstack([coords[nalign:], coords[:nalign]])
    assert coords2.shape == coords1.shape

    stub1 = geom_util.stub_from_three_points(coords1[0], coords1[1], coords1[2])
    coords1 = superimpose_coords(coords2, coords1) # fix,mov
    stub2 = geom_util.stub_from_three_points(coords1[0], coords1[1], coords1[2])

    center, n, t, theta = geom_util.get_stub_transform_data(stub1, stub2)

    # center stub at the midpoint of the line connecting the COMs
    #  of the two sets of points
    v = 0.5*(acom+bcom)

    x = n # symmetry axis
    if point_towards is not None:
        if (point_towards-v).dot(x) < 0:
            if verbose:
                print('flip symmetry axis:', (point_towards-v).dot(x))
            x *= -1.0

    z = bcom-acom # vector between COMs
    z -= x * x.dot(z)
    z /= norm(z)
    y = np.cross(z, x)

    if verbose:
        print("get_symmetry_stub_from_coords: theta= ", theta,
              " n: ", n, " nalign: ", nalign)

    return {'axes': np.stack([x,y,z]), 'origin': v}

def get_symmetry_stub_from_positions(
        poslist,
        pose,
        point_towards=None, # disambiguate symmetry axis (ie x) direction
        verbose=False,
):
    '''

    The idea is that the poslist[:L/2] residues and poslist[L/2:] residues
    are related by roughly 180 degree rotation

    align poslist[:] rsds onto poslist[L/2:]+poslist[:L/2] rsds,
       using CA positions, and skipping zeros in poslist in either half

    if acom = COM(poslist[:L/2]) and bcom = COM(poslist[L/2:]), where
     COM=center of mass

    the stub center is (acom+bcom)/2
    the stub x-axis is parallel with the symmetry axis
    the stub z-axis is (bcom-acom) but with unit length
    and y = z cross x

    '''

    return get_symmetry_stub_from_coords(
        pose['ca_coords'][poslist], point_towards=point_towards,
    )
