# import numpy as np
import sys
# from os import system
import os.path
from os import system, popen
from os.path import exists
from pathlib import Path
import pandas as pd
import numpy as np

from . import superimpose
from . import util
from . import blast
from . import sequtil
from . import pdblite

from .blast import path_to_blast_executables
from .util import path_to_db
from .sequtil import ALL_GENES_GAP_CHAR

NUM_MHC_CORE_POSITIONS = 12


## 1-indexed
## numbered wrt the sequence in 3pqyA.fasta ie class1_template_seq below
## to get 3pqy PDB numbers just add 1
## 4...10 are upward facing in the N-terminal central strand
## 23...25 are upward facing in the neighboring N-terminal strand
## 94...100 are upward facing in the C-terminal central strand
## 113.115 are upward facing in the neighboring C-terminal
##
class1_template_core_positions_1indexed = [4,  6,  8,  10,  23,  25,
                                           94, 96, 98, 100, 113, 115]

class1_template_core_positions_0indexed = [
    x-1 for x in class1_template_core_positions_1indexed]

class1_template_seq = 'PHSMRYFETAVSRPGLEEPRYISVGYVDNKEFVRFDSDAENPRYEPRAPWMEQEGPEYWERETQKAKGQEQWFRVSLRNLLGYYNQSAGGSHTLQQMSGCDLGSDWRLLRGYLQFAYEGRDYIALNEDLKTWTAADMAAQITRRKWEQSGAAEHYKAYLEGECVEWLHRYLKNGNATLLRTDSPKAHVTHHPRSKGEVTLRCWALGFYPADITLTWQLNGEELTQDMELVETRPAGDGTFQKWASVVVPLGKEQNYTCRVYHEGLPEPLTLRWEP'

class2_alfas_positions_0indexed = {
    'A': [2, 4, 7, 9, 18, 20],
    'B': [2, 4, 6, 8, 21, 23], # 8 is disulfide posn
}

def get_mhc_core_positions_class1(seq):
    ''' These are 0-indexed positions (unlike previous tcrdock)

    will have -1 if there's a parse fail
    '''
    al = sequtil.blosum_align(class1_template_seq, seq)

    core_positions = [
        al.get(pos, -1) for pos in class1_template_core_positions_0indexed
    ]

    assert -1 not in core_positions # probably should handle this better
    return core_positions


def get_mhc_core_positions_class2(
        aseq, # chain 1 ie alpha
        bseq, # chain 2 ie beta
):
    ''' these are 0-indexed positions!!! (unlike previous tcrdock)

    will have -1 if there's a parse fail
    '''
    offset = 0
    core_positions = []

    for ab, seq in zip('AB',[aseq,bseq]):
        dbfile = str(path_to_db / f'both_class_2_{ab}_chains_v2.fasta')
        hits = blast.blast_sequence_and_read_hits(seq, dbfile)
        hit = hits.iloc[0]
        blast_align = blast.setup_query_to_hit_map(hit)
        if hit.pident<99.99:
            print('get_mhc_core_positions_class2:', ab, hit.saccver, hit.pident)

        alfas = sequtil.mhc_class_2_alfas[ab][hit.saccver]
        hitseq = alfas.replace(ALL_GENES_GAP_CHAR, '')
        blast_hitseq = hit.sseq.replace('-','') # debug
        assert hitseq.index(blast_hitseq) == hit.sstart-1 # debug

        # make an alignment from alfas to seq
        alfas2hitseq = {i:i-alfas[:i].count(ALL_GENES_GAP_CHAR)
                        for i,a in enumerate(alfas) if a != ALL_GENES_GAP_CHAR}
        hitseq2seq = {j:i for i,j in blast_align.items()}

        for pos in class2_alfas_positions_0indexed[ab]:
            hitseq_pos = alfas2hitseq[pos]
            if hitseq_pos in hitseq2seq:
                core_positions.append(int(offset + hitseq2seq[hitseq_pos]))
            else:
                core_positions.append(-1)

        offset += len(seq)
    assert -1 not in core_positions # probably should handle this better
    return core_positions


def _setup_class_2_alfas_blast_dbs():
    ''' Just called once during setup
    '''

    makeblastdb_exe = str(path_to_blast_executables / 'makeblastdb')

    for ab in 'AB':
        alfas_fname = str(path_to_db / f'both_class_2_{ab}_chains_v2.alfas')
        fasta_fname = alfas_fname[:-5]+'fasta'

        alfas = sequtil.read_fasta(alfas_fname)
        out = open(fasta_fname, 'w')
        for name,alseq in alfas.items():
            seq = alseq.replace(ALL_GENES_GAP_CHAR, '')
            out.write(f'>{name}\n{seq}\n')
        out.close()

        # format the db
        cmd = f'{makeblastdb_exe} -in {fasta_fname} -dbtype prot'
        print(cmd)
        system(cmd)



def get_mhc_stub(
        pose,
        tdinfo = None,
        mhc_core_positions = None,
):
    ''' chain 1 (or 1 and 2) should be the mhc chain
    chain -3 should be the peptide chain if TCR else chain -1 (pMHC only)

    the mhc_core_positions (or tdinfo.mhc_core) are half on one side of the
    beta sheet center and half on the other side, and
    mhc_core_positions[:6]

    the stub center is at the midpoint of the two centers of mass
    the stub x-axis is parallel with the symmetry axis of the approximate
        180 degree rotation relating one side of the beta sheet to the other
        and points toward the peptide
    the stub z-axis points from one half of the beta sheet to the other

    (see superimpose.get_symmetry_stub_from_positions)

    returns a stub aka
    dict {'axes':numpy array with axes as ROW vectors,
          'origin': vector}

    '''

    if mhc_core_positions is None:
        if tdinfo is not None:
            mhc_core_positions = tdinfo.mhc_core
        else:
            print('get_mhc_stub needs either tdinfo or mhc_core_positions')
            sys.exit()

    num_chains = len(pose['chains'])
    if num_chains in [4,5]:
        pep_chain = num_chains-3
    else:
        # huh, maybe mhc only?
        assert num_chains in [2,3]
        pep_chain = num_chains-1

    # we may need to flip
    #current_x = stub.rotation().col_x()
    # was hardcoding pep_chain to 2 here...
    chainbounds = pose['chainbounds']
    pep_positions = list(range(chainbounds[pep_chain], chainbounds[pep_chain+1]))
    pep_centroid = np.mean(pose['ca_coords'][pep_positions], axis=0)

    stub = superimpose.get_symmetry_stub_from_positions(
        mhc_core_positions, pose, point_towards=pep_centroid)

    assert (pep_centroid - stub['origin']).dot(stub['axes'][0]) > 0
    return stub


def orient_pmhc_pose(
        pose,
        mhc_core_positions = None,
        mhc_class = None,
        tdinfo = None, # TCRdockInfo
):
    ''' chain 1 should be the mhc chain
    chain 2 should be the peptide chain
    '''
    if mhc_core_positions is None:
        if tdinfo is None:
            cs = pose['chainseq'].split('/')
            if mhc_class == 1:
                mhc_core_positions = get_mhc_core_positions_class1(cs[0])
            else:
                assert mhc_class == 2
                mhc_core_positions = get_mhc_core_positions_class2(cs[0], cs[1])
        else:
            mhc_core_positions = tdinfo.mhc_core

    stub = get_mhc_stub(pose, mhc_core_positions=mhc_core_positions)

    R = stub['axes']
    v = -1. * R @ stub['origin']

    pose = pdblite.apply_transform_Rx_plus_v(pose, R, v)

    return pose


def make_sorting_tuple(allele):
    ''' Helper function for choosing among equal blast hits
    '''
    if '*' not in allele:
        assert ':' not in allele
        return (allele,0)
    else:
        pref, suf = allele.split('*') # just one star
        if suf[-1].isdigit():
            suf = [int(x) for x in suf.split(':')]
        else:
            suf = [int(x) for x in suf[:-1].split(':')]+[suf[-1]]

        return tuple([pref]+suf)



def get_mhc_allele(seq, organism, return_identity=False):
    if organism == 'human':
        dbfile = (util.path_to_db /
                  'hla_prot_plus_trimmed_minus_funny_w_CD1s_nr_v2.fasta')
    else:
        assert organism == 'mouse'
        dbfile = (util.path_to_db /
                  'mhc_pdb_chains_mouse_reps_reformat.fasta')


    hits = blast.blast_sequence_and_read_hits(
        seq, dbfile, num_alignments=3)
    top_bitscore = max(hits.bitscore)
    top_hits = hits[hits.bitscore == top_bitscore].reset_index()
    if top_hits.shape[0]>1:# ties
        #print('ties:', top_hits.shape[0])
        sortl = [(make_sorting_tuple(a), a, i) for i,a in enumerate(top_hits.saccver)]
        sortl.sort()
        #print('sorted:', sortl)
        ind = sortl[0][-1]
        hit = top_hits.iloc[ind]
        assert hit.saccver == sortl[0][1]
    else:
        hit = top_hits.iloc[0]
    mhc = hit.saccver
    if organism == 'mouse' and len(mhc) == 3 and mhc[1] == '-':
        mhc = f'H2{mhc[0]}{mhc[2].lower()}'

    if return_identity:
        return mhc, hit.pident
    else:
        return mhc




