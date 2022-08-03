# import numpy as np
# import sys
# from os import system
import os.path
import itertools as it
from os import system, popen
from os.path import exists
from pathlib import Path
import pandas as pd
import numpy as np
from . import superimpose
from . import util


# moved to tcrdist.parsing and changed returnvals slightly
# def parse_tcr_sequence(
#         organism,
#         ab,
#         chainseq,
# ):


def get_tcr_stub(
        pose,
        tdinfo, # TCRdockInfo
):
    '''
    intuition: the stub is determined by the superposition that fits
    the alpha chain of the tcr onto the beta chain

    the stub center is at the midpoint of the two centers of mass
    the stub x-axis is parallel with the symmetry axis of the approximate
        180 degree rotation relating the two chains
        and points toward the CDR loops
    the stub z-axis points from the alpha chain to the beta chain

    uses tdinfo.tcr_core and tdinfo.tcr_cdrs

    '''
    cdr_positions = list(it.chain(*tdinfo.tcr_cdrs))
    cdr_centroid = np.mean(pose['ca_coords'][cdr_positions], axis=0)

    stub = superimpose.get_symmetry_stub_from_positions(
        tdinfo.tcr_core, pose, point_towards=cdr_centroid)

    assert (cdr_centroid - stub['origin']).dot(stub['axes'][0]) > 0

    return stub



######################################################################################88
######################################################################################88
## stuff below here has not been refactored yet
## I think it's mostly about the single-chain (alpha or beta) stubs
######################################################################################88
######################################################################################88



avg_core_coords = {}

def _load_avg_core_coords(organism):
    ''' returns [ alpha_coords, beta_coords ]
    alpha_coords and beta_coords are Vectors ie utility::vector1< Vector >
    '''
    global avg_core_coords
    if organism not in avg_core_coords:
        coords = []
        for ab in 'AB':
            coordsfile = util.path_to_db / f'tcr_core_{ab}_{organism}.txt'
            print('reading avg_core_coords from', coordsfile)
            assert exists(coordsfile)
            coords.append(make_Vectors(np.loadtxt(coordsfile)))
        avg_core_coords[organism] = coords
    return avg_core_coords[organism]

tcr_stub_transforms_cached = None
def _load_tcr_stub_transforms():
    global tcr_stub_transforms_cached
    if tcr_stub_transforms_cached is None:
        filename = util.path_to_db / 'tcr_stub_transforms.txt'
        assert exists(filename)
        A = np.loadtxt(filename)
        assert A.shape == (8,3)
        aR = Matrix.rows(*A[0:3].ravel())
        bR = Matrix.rows(*A[4:7].ravel())
        av = Vector(*A[3])
        bv = Vector(*A[7])
        tcr_stub_transforms_cached = (aR, av, bR, bv)
    return tcr_stub_transforms_cached

def get_tcr_chain_stubs(
        pose,
        tdinfo, # TCRdockInfo
        organism, # need this since mouse and human tcrs might be different??
):
    ''' Returns astub, bstub

    defined by superposition of avg chain coords onto the core positions
    for the pose
    '''
    from pyrosetta.rosetta.core.chemical import aa_cys
    from pyrosetta.rosetta.core.kinematics import Stub
    NUM_CORE = 13
    # these are indices into the 13 core_positions
    #  1 and 12 are disulfide positions
    # stub should be centered at 1, with x-axis from 1 to 12,
    #  y-axis from 1 to 0, z perp
    CENTER_IDX, X_IDX, Y_IDX = 1, 12, 0
    avg_core_coords = _load_avg_core_coords(organism)
    core_stubs = []
    for ch in range(2):
        core_positions = tdinfo.tcr_core[ch*NUM_CORE:(ch+1)*NUM_CORE]
        assert pose.residue(core_positions[CENTER_IDX]).aa() == aa_cys
        assert pose.residue(core_positions[     X_IDX]).aa() == aa_cys
        core_coords = [pose.residue(x).xyz("CA")
                       for x in core_positions]
        mov_coords = make_Vectors(avg_core_coords[ch]) # makes a copy
        superimpose.superimpose_coords(core_coords, mov_coords)
        # agh this is so painful: vector1 is 1-indexed!
        core_stubs.append(
            Stub(mov_coords[CENTER_IDX+1],
                 mov_coords[X_IDX     +1],
                 mov_coords[CENTER_IDX+1],
                 mov_coords[Y_IDX     +1]));

    return core_stubs[0], core_stubs[1]

def get_tcr_chain_stubs_from_ca_coords(
        ca_coords_in,
        tdinfo, # TCRdockInfo
        organism, # need this since mouse and human tcrs might be different??
):
    ''' Returns astub, bstub

    defined by superposition of avg chain coords onto the core positions
    for the pose
    '''
    from pyrosetta.rosetta.core.kinematics import Stub
    NUM_CORE = 13

    ca_coords = make_Vectors(ca_coords_in)
    # these are indices into the 13 core_positions
    #  1 and 12 are disulfide positions
    # stub should be centered at 1, with x-axis from 1 to 12,
    #  y-axis from 1 to 0, z perp
    CENTER_IDX, X_IDX, Y_IDX = 1, 12, 0
    avg_core_coords = _load_avg_core_coords(organism)
    core_stubs = []
    for ch in range(2):
        # these core_positions are 1-indexed (pose numbering)
        core_positions = tdinfo.tcr_core[ch*NUM_CORE:(ch+1)*NUM_CORE]
        core_coords = [ca_coords[x] for x in core_positions]
        mov_coords = make_Vectors(avg_core_coords[ch]) # makes a copy
        superimpose.superimpose_coords(core_coords, mov_coords)
        # agh this is so painful: vector1 is 1-indexed!
        core_stubs.append(
            Stub(mov_coords[CENTER_IDX+1],
                 mov_coords[X_IDX     +1],
                 mov_coords[CENTER_IDX+1],
                 mov_coords[Y_IDX     +1]))

    return core_stubs[0], core_stubs[1]

def get_tcr_chain_stubs_from_core_coords(
        core_coords, # numpy array
):
    ''' Returns dict with the stubs:

    return {
        'A':astub,
        'B':bstub,
        'AB_A':pstub_A,
        'AB_B':pstub_B,
    }

    defined by superposition of avg chain coords onto the core positions
    for the pose
    '''

    from pyrosetta.rosetta.core.kinematics import Stub
    NUM_CORE = 13

    assert core_coords.shape == (2*NUM_CORE, 3)


    # these are 0-indices into the 13 core_positions
    #  1 and 12 are disulfide positions
    # stub should be centered at 1, with x-axis from 1 to 12,
    #  y-axis from 1 to 0, z perp
    CENTER_IDX, X_IDX, Y_IDX = 1, 12, 0
    avg_core_coords = _load_avg_core_coords('human') # hardcode organism doesnt matter

    core_stubs = []
    for ch in range(2):
        ch_core_coords = core_coords[ch*NUM_CORE:(ch+1)*NUM_CORE]
        mov_coords = make_Vectors(avg_core_coords[ch]) # makes a copy
        superimpose.superimpose_coords(ch_core_coords, mov_coords)
        # agh this is so painful: vector1 is 1-indexed!
        core_stubs.append(
            Stub(mov_coords[CENTER_IDX+1],
                 mov_coords[X_IDX     +1],
                 mov_coords[CENTER_IDX+1],
                 mov_coords[Y_IDX     +1]))

    astub, bstub = core_stubs

    # now create pseudo-paired stubs based on these chain stubs
    aR, av, bR, bv = _load_tcr_stub_transforms()

    pstub_A = Stub()
    pstub_A.M = astub.M * aR
    pstub_A.v = astub.local2global(av)

    pstub_B = Stub()
    pstub_B.M = bstub.M * bR
    pstub_B.v = bstub.local2global(bv)

    return {
        'A':astub,
        'B':bstub,
        'AB_A':pstub_A,
        'AB_B':pstub_B,
    }
