import os
import sys
from sys import exit
import copy
import itertools as it
from pathlib import Path
from os.path import exists, isdir
import design_paths
design_paths.setup_import_paths()
import tcrdock as td2
from tcrdock.tcrdist.amino_acids import amino_acids
from tcrdock.docking_geometry import compute_docking_geometries_distance_matrix
import pandas as pd
import numpy as np
import random
from os import system, popen, mkdir
from glob import glob
from collections import Counter, OrderedDict, namedtuple
import scipy
import json


all_template_poses = {} # global dict to avoid reloading...
def get_const_pose_and_tdinfo(fname):
    global all_template_poses
    if fname not in all_template_poses:
        pose = td2.pdblite.pose_from_pdb(fname)
        tdifile = fname+'.tcrdock_info.json'
        if exists(tdifile):
            with open(tdifile, 'r') as f:
                tdinfo = td2.tcrdock_info.TCRdockInfo().from_string(f.read())
        else:
            tdinfo = None
        all_template_poses[fname] = (pose, tdinfo)

    return all_template_poses[fname] # NOT making a copy!

rmsd_atoms = [' N  ', ' CA ', ' C  ', ' O  ']

def get_designable_positions(
        alignstring=None,
        row=None,
        extend_flex=1,
        nres=None,
        reverse=False,
        num_contigs=2, # for debugging
):
    ''' the unaligned position plus extend_flex rsds on either side of each contig
    '''
    if hasattr(row, 'designable_positions'):
        posl = [int(x) for x in row.designable_positions.split(',')]
        return sorted(posl) # want sorted!

    if alignstring is None:
        alignstring = row.template_0_target_to_template_alignstring

    align = dict(tuple(map(int, x.split(':'))) for x in alignstring.split(';'))
    if reverse:
        align = {y:x for x,y in align.items()}
    fixed_posl = sorted(align.keys())
    if nres is None: # assume no terminal designable positions
        nres = max(fixed_posl)+1
    flex_posl = [x for x in range(nres) if x not in fixed_posl]
    if extend_flex:
        new_flex_posl = set()
        for pos in flex_posl:
            new_flex_posl.update([
                pos+i for i in range(-extend_flex, extend_flex+1)])
        assert len(new_flex_posl) == len(flex_posl) + 2*extend_flex*num_contigs
        flex_posl = sorted(new_flex_posl)
    return flex_posl

def get_rmsd_coords(pose, flex_posl):
    '''
    '''
    coords = pose['coords']
    resids = pose['resids']

    return np.stack([coords[resids[i]][a] for i in flex_posl for a in rmsd_atoms])

def filled_in_alignment(align_in):
    'returns a new dict, does not change input'
    align = copy.deepcopy(align_in)
    nres = max(align.keys())+1
    nat_nres = max(align.values())+1
    #new_align = deepcopy(align)
    gap_starts = [i for i in align if i+1 not in align and i+1<nres]
    gap_stops  = [i for i in align if i-1 not in align and i>0]
    assert len(gap_starts) == 2 and len(gap_stops) == 2
    for start, stop in zip(sorted(gap_starts), sorted(gap_stops)):
        nstart, nstop = align[start], align[stop]
        mingap = min(stop-start-1, nstop-nstart-1)
        nterm, cterm = mingap//2, mingap-mingap//2
        for i in range(nterm):
            align[start+1+i] = nstart+1+i
        for i in range(cterm):
            align[stop-1-i] = nstop-1-i
    #print(len(align), nres, nat_nres)
    #assert len(align) == min(nres, nat_nres)
    assert len(set(align.values())) == len(align)
    return align


def compute_stats(
        targets,
        extend_flex=1,
):
    ''' returns targets with stats like peptide_loop_pae, rmsds, recovery filled in
    (assuming the template pdbfiles are the native structures)
    '''
    required_cols = ('chainseq template_0_template_pdbfile '
                     ' template_0_target_to_template_alignstring '
                     ' model_pdbfile model_plddtfile model_paefile'.split())
    for col in required_cols:
        assert col in targets.columns

    dfl = []
    for _,l in targets.iterrows():
        nres = len(l.chainseq.replace('/',''))
        alignstring = l.template_0_target_to_template_alignstring
        nat_pose, tdinfo = get_const_pose_and_tdinfo(l.template_0_template_pdbfile)
        a,b,c = nat_pose['chainbounds'][:3] # a = 0
        nat_sequence = nat_pose['sequence']
        sequence = l.chainseq.replace('/','')

        plddts = np.load(l.model_plddtfile)[:nres]
        paes = np.load(l.model_paefile)[:nres,:][:,:nres]

        align = dict(tuple(map(int,x.split(':'))) for x in alignstring.split(';'))
        align_rev = {y:x for x,y in align.items()}
        mod_flex_posl = get_designable_positions(
            alignstring=alignstring, extend_flex=extend_flex)
        nat_flex_posl = get_designable_positions(
            alignstring=alignstring, extend_flex=extend_flex, reverse=True)

        if tdinfo is None:
            nat_cdr3_bounds = [[0,1],[2,3]] # hacking
        else:
            nat_cdr3_bounds = [tdinfo.tcr_cdrs[3], tdinfo.tcr_cdrs[7]]
        mod_cdr3_bounds = [[align_rev[x[0]], align_rev[x[1]]] for x in nat_cdr3_bounds]

        nat_cdr3s, mod_cdr3s = [], []
        for start,stop in nat_cdr3_bounds:
            nat_cdr3s.append(nat_sequence[start:stop+1])
            mod_cdr3s.append(sequence[align_rev[start]:align_rev[stop]+1])

        # actually two loops:
        loop_seq = ''.join(sequence[x] for x in mod_flex_posl)
        wt_loop_seq = ''.join(nat_sequence[x] for x in nat_flex_posl)


        # rmsds
        pose = td2.pdblite.pose_from_pdb(l.model_pdbfile)
        align_full = filled_in_alignment(align)
        mod_coords = get_rmsd_coords(
            pose, [x for x in mod_flex_posl if x in align_full])
        nat_coords = get_rmsd_coords(
            nat_pose,
            [align_full[x] for x in mod_flex_posl if x in align_full])
        natoms = mod_coords.shape[0]
        assert nat_coords.shape == (natoms, 3)
        cdr3a_flex_coords_len = sum(
            x in align_full and x in mod_flex_posl for x in range(*mod_cdr3_bounds[0]))
        cdr3b_flex_coords_len = sum(
            x in align_full and x in mod_flex_posl for x in range(*mod_cdr3_bounds[1]))
        #print('cdr3a_flex_coords_len:', cdr3a_flex_coords_len,
        #      cdr3b_flex_coords_len, natoms//len(rmsd_atoms))
        assert ((cdr3a_flex_coords_len + cdr3b_flex_coords_len)*len(rmsd_atoms) ==
                natoms)

        nat_mhc_coords = nat_pose['ca_coords'][:b]
        mhc_coords = pose['ca_coords'][:b]

        R, v = td2.superimpose.superimposition_transform(
            nat_mhc_coords, mhc_coords)

        mod_coords = (R@mod_coords.T).T + v
        rmsd = np.sqrt(np.sum((nat_coords-mod_coords)**2)/natoms)
        split = cdr3a_flex_coords_len * len(rmsd_atoms)
        rmsda = np.sqrt(np.sum((nat_coords[:split]-mod_coords[:split])**2)/split)
        rmsdb = np.sqrt(np.sum((nat_coords[split:]-mod_coords[split:])**2)/
                        (natoms-split))


        outl = l.copy()
        outl['loop_plddt'] = plddts[mod_flex_posl].mean()
        outl['loop_rmsd'] = rmsd
        outl['aloop_rmsd'] = rmsda
        outl['bloop_rmsd'] = rmsdb
        outl['loop_seq'] = loop_seq
        outl['wt_loop_seq'] = wt_loop_seq
        outl['cdr3a'] = mod_cdr3s[0]
        outl['wt_cdr3a'] = nat_cdr3s[0]
        outl['ashift'] = len(mod_cdr3s[0])-len(nat_cdr3s[0])
        outl['cdr3b'] = mod_cdr3s[1]
        outl['wt_cdr3b'] = nat_cdr3s[1]
        outl['bshift'] = len(mod_cdr3s[1])-len(nat_cdr3s[1])
        outl['peptide'] = l.chainseq.split('/')[1]
        outl['wt_peptide'] = nat_pose['chainseq'].split('/')[1]
        outl['peptide_plddt'] = plddts[b:c].mean()
        outl['peptide_loop_pae'] = 0.5*(
            paes[b:c,:][:,mod_flex_posl].mean() +
            paes[mod_flex_posl,:][:,b:c].mean())

        dfl.append(outl)


    targets = pd.DataFrame(dfl)

    return targets



mhc_seq2core_positions = {}
def get_mhc_core_positions_0x_cached(mhc_class, mhc_seq, verbose=True):
    global mhc_seq2core_positions
    # if mhc_class==2 mhc_seq is a tuple
    if (mhc_class, mhc_seq)  not in mhc_seq2core_positions:
        if verbose:
            print('get mhc core:', mhc_class, mhc_seq)
        if mhc_class==1:
            posl = td2.mhc_util.get_mhc_core_positions_class1(mhc_seq)
        else:
            posl = td2.mhc_util.get_mhc_core_positions_class2(*mhc_seq)
        mhc_seq2core_positions[(mhc_class,mhc_seq)] = posl

    return mhc_seq2core_positions[(mhc_class, mhc_seq)]

def get_model_tdinfo(
        organism,
        mhc_class,
        mhc_allele,
        cb_seq,
        va,
        ja,
        cdr3a,
        vb,
        jb,
        cdr3b,
        verbose = True,
):
    ''' return tdinfo

    meant to be faster than starting from sequences

    but falls back on slow parsing if the tcr framework sequences dont match
    expectation

    '''
    from tcrdock.sequtil import ALL_GENES_GAP_CHAR
    from tcrdock.tcrdist.all_genes import all_genes
    from tcrdock.tcrdist.parsing import parse_tcr_sequence

    core_len = 13 # tcr core len

    sequence = cb_seq.replace('/','')
    nres = len(sequence)
    *mhc_seqs, pep_seq, tcra_seq, tcrb_seq = cb_seq.split('/')
    chainlens = [len(x) for x in cb_seq.split('/')]
    assert mhc_class == len(mhc_seqs)

    mhc_len = sum(chainlens[:-3])
    pmhc_len = sum(chainlens[:-2])
    pmhc_tcra_len = sum(chainlens[:-1])

    tcra_prefix = td2.sequtil.get_v_seq_up_to_cys(organism, va)[:-1] # dont require Cys
    tcrb_prefix = td2.sequtil.get_v_seq_up_to_cys(organism, vb)[:-1]

    if tcra_seq.startswith(tcra_prefix) and tcrb_seq.startswith(tcrb_prefix):
        # yay! we got a match to 'clean' model tcr seqs up to the Cys
        tcr_core_positions = (
            [x+pmhc_len      for x in td2.sequtil.get_core_positions_0x(organism, va)]+
            [x+pmhc_tcra_len for x in td2.sequtil.get_core_positions_0x(organism, vb)])

        tcr_cdrs = []
        for v, seq, cdr3, offset in [[va, tcra_seq, cdr3a, pmhc_len],
                                     [vb, tcrb_seq, cdr3b, pmhc_tcra_len]]:
            g = all_genes[organism][v]
            for gapcdr in g.cdrs[:3]+[cdr3]:
                cdr = gapcdr.replace(ALL_GENES_GAP_CHAR,'')
                if cdr != cdr3 and cdr in cdr3:
                    # special wonky case
                    start = seq[:seq.index(cdr3)].index(cdr) + offset
                else:
                    assert seq.count(cdr) == 1
                    start = seq.index(cdr) + offset
                tcr_cdrs.append([start, start+len(cdr)-1])
    else:
        # have to do it the slow and painful way
        print('WARNING get_model_tdinfo: slow and painful tcr parsing')
        aparse = parse_tcr_sequence(organism, 'A', tcra_seq)
        bparse = parse_tcr_sequence(organism, 'B', tcrb_seq)
        assert (aparse is not None) and (bparse is not None),\
            'get_model_tdinfo tcr seq parsing failed!'
        tcr_core_positions = (
            [x+pmhc_len      for x in aparse['core_positions']]+
            [x+pmhc_tcra_len for x in bparse['core_positions']]
        )
        tcr_cdrs = (
            [(x+pmhc_len     , y+pmhc_len     ) for x,y in aparse['cdr_loops']]+
            [(x+pmhc_tcra_len, y+pmhc_tcra_len) for x,y in bparse['cdr_loops']]
        )


    tcr_coreseq = ''.join(sequence[x] for x in tcr_core_positions)
    cys_seq = tcr_coreseq[1]+tcr_coreseq[12]+tcr_coreseq[14]+tcr_coreseq[25]
    if verbose or cys_seq != 'CCCC':
        print('tcr_coreseq:', cys_seq, tcr_coreseq[:core_len], tcr_coreseq[core_len:])


    mhc_seq = mhc_seqs[0] if mhc_class==1 else tuple(mhc_seqs)
    mhc_core_positions_0x = get_mhc_core_positions_0x_cached(mhc_class, mhc_seq,
                                                             verbose=False)

    tdinfo = td2.tcrdock_info.TCRdockInfo().from_dict(
        dict(organism = organism,
             mhc_class = mhc_class,
             mhc_allele = mhc_allele,
             mhc_core = mhc_core_positions_0x,
             pep_seq = pep_seq,
             tcr = ((va, ja, cdr3a), (vb, jb, cdr3b)),
             tcr_core = tcr_core_positions,
             tcr_cdrs = tcr_cdrs,
             valid = True,
        ))

    return tdinfo


def compute_docking_geometry_info(l, pose=None):
    ''' l is a Series

    returns None if failure

    otherwise returns dict with some info including dgeom_rmsd to prev dgeom if present
    '''

    # maybe compute docking geometry
    required_cols = ('organism mhc_class mhc chainseq va ja cdr3a vb jb cdr3b').split()
    if pose is None:
        required_cols.append('model_pdbfile')

    if not all(hasattr(l,x) for x in required_cols):
        print('WARNING compute_docking_geometry_info missing required cols:',
              [x for x in required_cols if not hasattr(l,x)])
        outl = None
    else:
        tdinfo = get_model_tdinfo(
            l.organism, l.mhc_class, l.mhc, l.chainseq,
            l.va, l.ja, l.cdr3a, l.vb, l.jb, l.cdr3b, verbose=False,
        )
        cbs = [0] + list(it.accumulate(len(x) for x in l.chainseq.split('/')))
        if pose is None:
            pose = td2.pdblite.pose_from_pdb(l.model_pdbfile)
        assert pose['sequence'] == l.chainseq.replace('/','')

        pose = td2.pdblite.set_chainbounds_and_renumber(pose, cbs)

        dgeom = td2.docking_geometry.get_tcr_pmhc_docking_geometry(pose, tdinfo)

        dgcols = 'torsion d tcr_unit_y tcr_unit_z mhc_unit_y mhc_unit_z'.split()

        outl = {}
        if hasattr(l, dgcols[0]):
            # already has docking geometry info, compute rmsd, save old info
            old_dgeom = td2.docking_geometry.DockingGeometry().from_dict(l)
            for col in dgcols:
                outl['old_'+col] = l[col]
            outl['dgeom_rmsd'] = compute_docking_geometries_distance_matrix(
                [dgeom], [old_dgeom], organism=l.organism)[0,0]

        for k,v in dgeom.to_dict().items():
            outl[k] = v

    return outl # either a dict or None

def compute_simple_stats(
        targets,
        extend_flex=1,
):
    ''' Not assuming a single 'native' template

    this assumes we can call:

      flex_posl = get_designable_positions(row=l, extend_flex=extend_flex)


    stats:

    loop_plddt
    loop_seq
    loop_seq2
    peptide
    peptide_plddt
    peptide_loop_pae
    pmhc_tcr_pae

    '''
    required_cols = ('chainseq model_plddtfile model_paefile'.split())
    for col in required_cols:
        assert col in targets.columns, f'Need {col} column in targets df'

    dfl = []
    for _, l in targets.iterrows():
        sequence = l.chainseq.replace('/','')
        nres = len(sequence)
        cbs = [0]+list(it.accumulate(len(x) for x in l.chainseq.split('/')))
        chain_number = np.zeros((nres,), dtype=int)
        for pos in cbs[1:-1]:
            chain_number[pos:] += 1
        num_chains = len(cbs)-1
        assert chain_number[0] == 0 and chain_number[-1] == num_chains-1

        assert l.mhc_class + 3 == num_chains
        nres_mhc, nres_pmhc = cbs[-3:-1]
        flex_posl = get_designable_positions(row=l, extend_flex=extend_flex)

        plddts = np.load(l.model_plddtfile)[:nres]
        paes = np.load(l.model_paefile)[:nres,:][:,:nres]

        # actually two loops:
        loop_seq = ''.join(sequence[x] for x in flex_posl)

        loop_seq2 = sequence[flex_posl[0]]
        for i,j in zip(flex_posl[:-1], flex_posl[1:]):
            if chain_number[i] != chain_number[j]:
                loop_seq2 += '/'
            loop_seq2 += sequence[j]

        outl = l.copy()
        outl['loop_plddt'] = plddts[flex_posl].mean()
        outl['loop_seq'] = loop_seq
        outl['loop_seq2'] = loop_seq2
        outl['peptide'] = l.chainseq.split('/')[-3]
        outl['peptide_plddt'] = plddts[nres_mhc:nres_pmhc].mean()
        outl['peptide_loop_pae'] = 0.5*(
            paes[nres_mhc:nres_pmhc,:][:,flex_posl].mean() +
            paes[flex_posl,:][:,nres_mhc:nres_pmhc].mean())
        outl['pmhc_tcr_pae'] = 0.5*(
            paes[:nres_pmhc,:][:,nres_pmhc:].mean() +
            paes[nres_pmhc:,:][:,:nres_pmhc].mean())

        # maybe compute docking geometry
        dginfo = compute_docking_geometry_info(l)
        if dginfo is not None:
            for k,v in dginfo.items():
                outl[k] = v
        dfl.append(outl)

    targets = pd.DataFrame(dfl)

    return targets



def add_info_to_rescoring_row(l, model_name, extend_flex=1):
    ''' l is a pandas Series
    uses chainseq, plddt-file, pae-file, template_0_alt_template_sequence
    YES extending of the loop definition by +/- extend_flex for the TCR
    '''
    #if model_name is None:
    #    model_name = l.model_name
    sequence = l.chainseq.replace('/','')
    nres = len(sequence)
    plddts = np.load(l[f'{model_name}_plddt_file'])[:nres]
    paes = np.load(l[f'{model_name}_predicted_aligned_error_file'])[:nres,:nres]

    gap_posl = [i for i,x in enumerate(l.template_0_alt_template_sequence)
                if x not in amino_acids]
    gap_starts = [i for i in gap_posl if i-1 not in gap_posl]
    gap_stops  = [i+1 for i in gap_posl if i+1 not in gap_posl]
    assert len(gap_starts) == 3 and len(gap_stops) == 3
    bounds = list(zip(gap_starts, gap_stops))
    pep_inds = list(range(*bounds[0]))
    peptide = ''.join(sequence[x] for x in pep_inds)
    peptide_plddt = plddts[pep_inds].mean()
    loop_inds = list(it.chain(range(*bounds[1]), range(*bounds[2])))
    if extend_flex:
        new_loop_inds = set()
        for pos in loop_inds:
            new_loop_inds.update([
                pos+i for i in range(-extend_flex, extend_flex+1)])
        assert len(new_loop_inds) == len(loop_inds) + 4*extend_flex
        loop_inds = sorted(new_loop_inds)
    #new_loop_inds = [i-1 for i in loop_inds] + [i+1 for i in loop_inds]
    loop_seq = ''.join(sequence[x] for x in loop_inds)
    loop_plddt = plddts[loop_inds].mean()
    peptide_loop_pae = 0.5 * (
        paes[pep_inds,:][:,loop_inds].mean() +
        paes[loop_inds,:][:,pep_inds].mean())
    mask_char = set(x for x in l.template_0_alt_template_sequence
                    if x not in amino_acids)
    assert len(mask_char) == 1
    mask_char = mask_char.pop()
    outl = l.copy()
    outl['model_name'] = model_name
    outl['peptide_plddt'] = peptide_plddt
    outl['peptide_len'] = len(pep_inds)
    outl['peptide'] = peptide
    outl['loop_plddt'] = loop_plddt
    outl['loop_len'] = len(loop_inds)
    outl['loop_seq'] = loop_seq
    outl['peptide_loop_pae'] = peptide_loop_pae
    outl['mask_char'] = mask_char

    return outl


if __name__ == '__main__':
    # testing

    print('hello world')

    tsvfile = ('/home/pbradley/csdat/tcrpepmhc/amir/'
               'run368run369_results_design_scores.tsv')
    targets = pd.read_table(tsvfile)

    targets = targets.head(10)

    targets = compute_simple_stats(targets)

    targets = compute_simple_stats(targets)

    if 0:
        for _,l in targets.iterrows():
            tdinfo = get_model_tdinfo(
                l.organism, l.mhc_class, l.mhc, l.chainseq,
                l.va, l.ja, l.cdr3a, l.vb, l.jb, l.cdr3b,
            )

            print(tdinfo)
