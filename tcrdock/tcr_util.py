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


