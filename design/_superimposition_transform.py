# hacky borrowing of some pyrosetta superposition code
#
import os.path
from os import system
from os.path import exists
from pathlib import Path
import pandas as pd
import pyrosetta
import numpy as np

#from . import geom_util

from pyrosetta.rosetta import core, protocols, numeric, basic, utility
from pyrosetta.rosetta.utility import \
    vector1_numeric_xyzVector_double_t as Vectors
from pyrosetta.rosetta.numeric import xyzVector_double_t as Vector
from pyrosetta.rosetta.numeric import xyzMatrix_double_t as Matrix
#from utility import vector1_bool as bools

def make_Vectors(vector_iterable):
    ''' Convert from an iterable to a utility::vector1< Vector >
    should work on numpy arrays of shape (*,3) as well as lists of Vector
    '''
    vectors = Vectors()
    for v in vector_iterable:
        vectors.append(Vector(*v)) # *v works for Vector and 1D numpy array
    return vectors


def superimposition_transform(fixcoords_in, movcoords_in, return_numpy=False):
    ''' fix_coords_in and movcoords_in are not modified
    they could be Vectors or lists, just need to loop over them

    fixcoords and movcoords are vector1_numeric_xyzVector_double_t
    ie they can be indexed [1] up to [len()]
    ie they are 1-indexed !!!

    returns rotation, translation that would orient movcoords_in onto fixcoords_in

    as in x--->rotation*x + translation

    '''

    fixcoords = make_Vectors(fixcoords_in)
    movcoords = make_Vectors(movcoords_in)

    natoms = len(fixcoords)
    assert natoms == len(movcoords)

    fix_com = Vector(0,0,0)
    mov_com = Vector(0,0,0)

    wts = utility.vector1_double()
    for i in range(1,natoms+1):
        fix_com += fixcoords[i]
        mov_com += movcoords[i]
        wts.append(1.0);

    fix_com /= natoms
    mov_com /= natoms

    R = Matrix()
    sigma3=0.0

    numeric.model_quality.findUU(fixcoords, movcoords, wts, natoms, R, sigma3)

    # indeed this confirms that findUU translates fixcoords and movcoords to have 0 COM
    # new_fix_com = Vector(0,0,0)
    # new_mov_com = Vector(0,0,0)

    # for i in range(1,natoms+1):
    #     new_fix_com += fixcoords[i]
    #     new_mov_com += movcoords[i]

    # new_fix_com /= natoms
    # new_mov_com /= natoms
    # print('COM after findUU:', new_fix_com, new_mov_com)
    v = fix_com - R*mov_com

    if return_numpy:
        return np.array(R), np.array(v)
    else:
        return R, v

def superimpose_coords(fixcoords, movcoords):
    ''' Move movcoords onto fixcoords
    Allow for these to be vector1<Vector> or list of Vector

    fixcoords could also be a numpy array of shape (N,3)
    but not movcoords (right now)
    '''

    N = len(fixcoords)
    assert N == len(movcoords)

    # this should not change fixcoords/movcoords
    R, translation = superimposition_transform(fixcoords, movcoords)

    inds = range(N) if type(movcoords) is list else range(1,N+1)

    for i in inds:
        movcoords[i] = R * movcoords[i] + translation

    # new_rmsd = np.sqrt(
    #     sum((x-y).length_squared() for x,y in zip(fixcoords, movcoords))/N)
    #print('superimpose_coords: rmsd:', new_rmsd)
    return

