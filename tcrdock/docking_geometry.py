import os.path
from os import system
from os.path import exists
from pathlib import Path
import pandas as pd
import numpy as np
import json
from scipy.cluster import hierarchy
from scipy.spatial import distance
from numpy.linalg import norm

from . import util
from . import geom_util
from . import superimpose
from . import mhc_util
from . import tcr_util

def dihedral_radians( p1, p2, p3, p4 ):
    assert p1.shape == (3,)
    assert p2.shape == (3,)
    assert p3.shape == (3,)
    assert p4.shape == (3,)
    # borrowed form Rosetta
    a = p2-p1
    a /= np.linalg.norm(a)
    b = p3-p2
    b /= np.linalg.norm(b)
    c = p4-p3
    c /= np.linalg.norm(c)

    x = -np.dot( a, c ) + ( np.dot( a, b ) * np.dot( b, c ) )
    y = np.dot( a, np.cross( b, c ) )
    angle = np.arctan2( y, x )
    return angle


class DockingGeometry():
    def __init__(self):
        pass

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

    def from_stubs(self, mhc_stub, tcr_stub):
        ''' stub is another word for coordinate frame (from protein folding)
        here a stub is a dictionary with two keys:
        'axes':3x3 orthogonal matrix with frame axes as ROW vectors
        'origin':3d vector, the origin of the coordinate frame
        all of those are in the global absolute reference frame in which the PDB
        coordinates are defined.
        '''
        from .geom_util import global2local

        #assert False
        self.torsion = dihedral_radians(
            mhc_stub['origin'] + mhc_stub['axes'][1],
            mhc_stub['origin'],
            tcr_stub['origin'],
            tcr_stub['origin'] + tcr_stub['axes'][2],
        )
        # want this in the range [0,2*pi) since most of the values
        # are around pi and the distn can get split between -pi and pi
        #
        self.torsion = (self.torsion + 2*np.pi)%(2*np.pi)

        # unit vector pointing toward TCR, in the MHC local coordsys
        #tcr_unit = mhc_stub.global2local(tcr_stub.v).normalized()
        tcr_unit = global2local(mhc_stub, tcr_stub['origin'])
        tcr_unit /= norm(tcr_unit)
        self.tcr_unit_y = tcr_unit[1]
        self.tcr_unit_z = tcr_unit[2]
        self.tcr_unit_x_is_negative = (tcr_unit[0]<0)


        # unit vector pointing toward MHC, in the TCR local coordsys
        mhc_unit = global2local(tcr_stub, mhc_stub['origin'])
        mhc_unit /= norm(mhc_unit)
        self.mhc_unit_y = mhc_unit[1]
        self.mhc_unit_z = mhc_unit[2]
        self.mhc_unit_x_is_negative = (mhc_unit[0]<0)

        self.d = norm(mhc_stub['origin'] - tcr_stub['origin'])

        if self.mhc_unit_x_is_negative or self.tcr_unit_x_is_negative:
            print("WARNING:: DockingGeometry::from_stubs negative unit x: ",
                  self.mhc_unit_x_is_negative, self.tcr_unit_x_is_negative)

        return self

    def to_dict(self):
        return dict(
            torsion=self.torsion,
            d=self.d,
            tcr_unit_y=self.tcr_unit_y,
            tcr_unit_z=self.tcr_unit_z,
            tcr_unit_x_is_negative=bool(self.tcr_unit_x_is_negative),
            mhc_unit_y=self.mhc_unit_y,
            mhc_unit_z=self.mhc_unit_z,
            mhc_unit_x_is_negative=bool(self.mhc_unit_x_is_negative),
            )

    def from_dict(self, D):
        # D could also be a pandas series
        # that's why we do the bool(...) below
        # otherwise we get a json error when we try to call __str__
        self.torsion = (D['torsion']+2*np.pi)%(2*np.pi)
        self.d = D['d']
        self.tcr_unit_y = D['tcr_unit_y']
        self.tcr_unit_z = D['tcr_unit_z']
        self.tcr_unit_x_is_negative = bool(D.get('tcr_unit_x_is_negative',False))
        self.mhc_unit_y = D['mhc_unit_y']
        self.mhc_unit_z = D['mhc_unit_z']
        self.mhc_unit_x_is_negative = bool(D.get('mhc_unit_x_is_negative',False))
        return self

    def to_string(self):
        return json.dumps(self.to_dict(), separators=(',', ':'))

    def from_string(self, info):
        return self.from_dict(json.loads(info))

def get_tcr_pmhc_docking_geometry(
        pose,
        tdinfo, # TCRdockInfo
):
    # get mhc stub
    mhc_stub = mhc_util.get_mhc_stub(pose, tdinfo=tdinfo)

    # get tcr stub
    tcr_stub = tcr_util.get_tcr_stub(pose, tdinfo)

    # check the flips
    return DockingGeometry().from_stubs(mhc_stub, tcr_stub)





def frame_from_docking_geometry(dgeom):
    ''' Assume the input stub (e.g. MHC stub) is located at the origin
    with canonical orientation
    Return a new stub (frame and center)
    thinking of the frames as column vectors
    '''
    from math import pi
    #RAD2DEG = 180. / math.pi

    from scipy.spatial.transform import Rotation


    # first get the downstream frame center in the correct location
    tcr_unit_x = np.sqrt(1 - dgeom.tcr_unit_y**2 - dgeom.tcr_unit_z**2)
    if dgeom.tcr_unit_x_is_negative:
        print('tcr_unit_x_is_negative')
        tcr_unit_x *= -1.
    tcr_center = dgeom.d * np.array(
        [tcr_unit_x, dgeom.tcr_unit_y, dgeom.tcr_unit_z])

    mhc_unit_x = np.sqrt(1 - dgeom.mhc_unit_y**2 - dgeom.mhc_unit_z**2)
    if dgeom.mhc_unit_x_is_negative:
        print('mhc_unit_x_is_negative')
        mhc_unit_x *= -1.
    mhc_center_in_tcr_frame = dgeom.d * np.array(
        [mhc_unit_x, dgeom.mhc_unit_y, dgeom.mhc_unit_z])

    # we want to rotate mhc_center about tcr_center so it aligns
    #  with the origin, applying that rotation to the std frame at tcr_center
    axis = np.cross(mhc_center_in_tcr_frame, -1*tcr_center)
    angle = np.arccos(np.dot(mhc_center_in_tcr_frame,-1*tcr_center)/dgeom.d**2)
    axis *= angle / np.linalg.norm(axis)

    # we are going to try thinking of the frames as column vectors
    frame = np.identity(3) # the tcr frame
    rot = Rotation.from_rotvec( axis ) # Rotation rotates row vectors...
    frame = rot.apply(frame.T).T
    # now the frame axes should have been rotated

    new_mhc_center = frame @ mhc_center_in_tcr_frame
    dev = np.linalg.norm(tcr_center + new_mhc_center)
    if dev > 0.1:
        print('WARNING::reconstruct_downstream_stub_from_docking_geometry '
              ' new_mhc_center dev:', dev)


    # ok so frame is rotated better but we might still need to rotate
    #  about the axis connecting tcr and mhc
    # from tcrmodel: it's the dihedral between mhc-y and tcr-z (duh?)
    #
    current_torsion = dihedral_radians(
        np.array([0,1.,0]), np.array([0.,0.,0.]),
        tcr_center, tcr_center+frame[:,2])
    axis = tcr_center / np.linalg.norm(tcr_center)
    angle = dgeom.torsion-current_torsion

    rot = Rotation.from_rotvec(angle * axis)

    frame = rot.apply(frame.T).T

    new_mhc_center = frame @ mhc_center_in_tcr_frame
    dev = np.linalg.norm(tcr_center + new_mhc_center)
    if dev > 0.1:
        print('WARNING::reconstruct_downstream_stub_from_docking_geometry'
              ' new_mhc_center2 dev:', dev)
    current_torsion = dihedral_radians(
        np.array([0,1.,0]), np.array([0.,0.,0.]),
        tcr_center, tcr_center+frame[:,2])
    dev = (current_torsion - dgeom.torsion)%(2*pi)
    if dev>pi:
        dev -= 2*pi
    if abs(dev)>0.01:
        print('WARNING:: reconstruct_downstream_stub_from_docking_geometry'
              ' torsion dev:', dev)

    return frame, tcr_center

def stub_from_docking_geometry(dgeom):
    # frame has the axes as column vectors; stub dict has row vectors
    frame, center = frame_from_docking_geometry(dgeom)
    #return Stub(Matrix.rows(*frame.ravel()), Vector(*center))
    return {'axes':frame.transpose(), 'origin':center}


def cdr_centroids_from_docking_geometry(
        dgeom,
        centroids_in_tcr_frame,
):
    assert centroids_in_tcr_frame.shape == (8,3)
    frame, center = frame_from_docking_geometry(dgeom)

    centroids = center + centroids_in_tcr_frame @ frame.T
    assert centroids.shape == (8,3)
    return centroids

cen_wts = {
    'AB': np.array([1,1,1,3,1,1,1,3.]),
    'A' : np.array([1,1,1,3,0,0,0,0.]),
    'B' : np.array([0,0,0,0,1,1,1,3.]),
}
def compute_centroids_distance_matrix(X, Y, chain='AB'):
    assert X.shape[1:] == (8,3)
    assert Y.shape[1:] == (8,3)
    #print 'compute D_XY', X.shape, Y.shape
    wts = cen_wts[chain]
    D = np.sqrt(np.sum(wts * np.sum(np.square(
        X[:,np.newaxis,...]-Y[np.newaxis,...]), axis=-1), axis=-1)/np.sum(wts))
    #print 'done computing D_XY', D.shape
    assert D.shape == ( X.shape[0], Y.shape[0] )
    return D

def compute_docking_geometries_distance_matrix(
        dgeoms1,
        dgeoms2,
        organism='human', # for the cdr centroids, doesnt really matter much at all
        verbose=False,
        chain='AB',
):
    cenfile = util.path_to_db / f'cdr_centroids_{organism}.txt'
    cdr_centroids_in_tcr_frame = np.loadtxt(cenfile)
    assert cdr_centroids_in_tcr_frame.shape == (8,3)
    if verbose:
        print('make centroids')
    centroids1 = np.stack(
        [cdr_centroids_from_docking_geometry(x, cdr_centroids_in_tcr_frame)
         for x in dgeoms1
        ])
    centroids2 = np.stack(
        [cdr_centroids_from_docking_geometry(x, cdr_centroids_in_tcr_frame)
         for x in dgeoms2
        ])
    if verbose:
        print('compute dists')
    return compute_centroids_distance_matrix(centroids1, centroids2, chain=chain)

def pick_docking_geometry_reps(
        organism,
        dgeoms,
        num_reps,
        tree_method = 'ward', # or 'average' ?
        outfile_prefix = None,
        leaf_label_prefixes = None,
):
    ''' Returns the rep 0-indices wrt dgeoms

    they are ordered in the leaf order for the tree, so neighbors are closer
    '''
    if len(dgeoms) < num_reps:
        return dgeoms[:], list(range(len(dgeoms)))

    D = compute_docking_geometries_distance_matrix(
        dgeoms, dgeoms, organism)

    #tree_optimal_ordering = False

    Z = hierarchy.linkage(
        distance.squareform(D,force='tovector'), method=tree_method)
    leaves = hierarchy.leaves_list(Z)

    clusters = hierarchy.fcluster(Z, num_reps, criterion='maxclust')

    rep_indices = []

    # the clusters seem to be in "leaf" order, starting at one end of the tree
    # and proceeding to the other...
    for c in sorted(set(clusters)):
        cmask = clusters==c
        cD = D[cmask,:][:,cmask]
        rep = np.argmin(cD.mean(axis=1))
        inds = np.nonzero(cmask)[0]
        rep_indices.append(inds[rep])

    new_rep_indices = [x for x in leaves if x in rep_indices]
    assert len(new_rep_indices) == len(rep_indices)
    #print('old:', rep_indices)
    #print('new:', new_rep_indices)
    # they seem to be the same
    rep_indices = new_rep_indices # confirm tree order

    if outfile_prefix is not None:
        if leaf_label_prefixes is not None:
            assert len(leaf_label_prefixes) == len(clusters)
            labels = [f'{x} R' if i in rep_indices else x
                      for i,x in enumerate(leaf_label_prefixes)]
        else:
            labels = [f'{c} R' if i in rep_indices else str(c)
                      for i,c in enumerate(clusters)]
        #import matplotlib
        #matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,11))
        hierarchy.dendrogram(
            Z, labels=labels,
            #leaf_font_size= 9, #leaf_font_size,
            orientation='right',# )#, ax=ax, link_color_func= lambda x:'k' )
        )
        pngfile = outfile_prefix+'_dgeom_tree.png'
        plt.savefig(pngfile, dpi=200)
        print('made:', pngfile)


    assert len(rep_indices) == num_reps
    return [dgeoms[x] for x in rep_indices], rep_indices




