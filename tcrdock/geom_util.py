import numpy as np
import sys
from scipy.spatial.transform import Rotation
import copy

# from os import system
#import os.path
#from os import system
#from os.path import exists
#from pathlib import Path
#import pandas as pd

def get_stub_transform_data(
        stub1,
        stub2,
        warn_debug=True
):
    from numpy.linalg import norm

    # R rotates stub1.M axis vectors into stub2.M axis vectors
    # stub axes are stored as row vectors ie
    # x-axis vec = stub['axes'][0], y-axis vec = stub['axes'][1], etc
    R = stub2['axes'].T @ stub1['axes']
    r = Rotation.from_matrix(R)
    theta_n = r.as_rotvec()
    #print('theta_n:', theta_n)

    theta = norm(theta_n)
    n = theta_n/theta
    #print('n:', norm(n), n)

    #n = theta_n.normalized()
    #theta = theta_n.length()
    #assert abs(n.length()-1.)<1e-3
    assert theta > -1e-3
    assert theta < np.pi + 1e-3
    v1 = stub1['origin']
    v2 = stub2['origin']
    #print('trans:', (v2-v1).dot(n))
    t = n * (v2-v1).dot(n)
    v2p = v2-t
    y = norm(v1-v2p)/2
    if theta < np.pi/2.: # problems if theta is close to 0? yep
        x = y/np.tan(theta/2)
    else:
        # u = x/y
        u2 = max(0., -1. + 1./np.sin(theta/2)**2)
        x = y*np.sqrt(u2)
    #print('xy:', x, y)
    midpoint = 0.5*(v1+v2p)
    ihat = (v2p-v1)/norm(v2p-v1)
    khat = n
    jhat = np.cross(khat, ihat)

    center = x *jhat + midpoint

    dev = norm(v2 - (R@( v1-center) + t + center ))
    #print('theta:', theta, 'dev:', dev)

    if warn_debug and theta > 1e-2 and dev>1e-1:
        print(f'WARNING: get_stub_transform_data: dev: {dev} theta: {theta}')

    return center, n, t, theta

## these are modeled on the Rosetta functions
def stub_from_four_points(center, a, b, c):

    from numpy.linalg import norm

    stub = {'origin':copy.deepcopy(center)}

    # x-axis goes from b to a
    # y-axis goes from b toward c in the abc plane

    ihat = (a-b)/norm(a-b)
    jhat = (c-b)
    jhat -= ihat * ihat.dot(jhat)
    jhat /= norm(jhat)
    khat = np.cross(ihat, jhat)

    stub['axes'] = np.stack([ihat,jhat,khat])

    return stub

def stub_from_three_points(a,b,c):
    return stub_from_four_points(a,a,b,c)

def global2local(stub, v):
    return stub['axes'].dot(v - stub['origin'])
