'''
'''

required_cols = ('targetid chainseq template_0_template_pdbfile model_pdbfile '
                 'template_0_target_to_template_alignstring'.split())

import argparse
parser = argparse.ArgumentParser(description="cluster loop designs")

parser.add_argument('--targets', required=True)
parser.add_argument('--outfile', required=True)
parser.add_argument('--mode', choices=['global','local'], default = 'global')
parser.add_argument('--num_runs', type=int, default=3, help='num af2 runs per design')

args = parser.parse_args()

import pandas as pd
targets = pd.read_table(args.targets)

for col in required_cols:
    assert col in targets.columns

assert targets.targetid.value_counts().max() == 1 # no duplicates


######################################################################################88
# more imports
import design_stats
import design_paths
design_paths.setup_import_paths()
import tcrdock as td2
from tcrdock.tcrdist.amino_acids import amino_acids
from os.path import exists
import itertools as it
import numpy as np

######################################################################################88

def get_rep_dgeoms_for_redocking(
        peptides_to_avoid,
        mhc_class,
        num_runs,
        min_mismatches=4,
):

    from tcrdock.sequtil import (ternary_info, BAD_DGEOM_PDBIDS,
                                  count_peptide_mismatches)
    dgeom_info = ternary_info[ternary_info.mhc_class == mhc_class]
    dgeom_info = dgeom_info[~dgeom_info.pdbid.isin(BAD_DGEOM_PDBIDS)].copy()
    dgeom_info['filt_peptide_mismatches'] = np.array(
        [min(count_peptide_mismatches(p, p2) for p in peptides_to_avoid)
         for p2 in dgeom_info.pep_seq])
    too_close_mask = dgeom_info.filt_peptide_mismatches < min_mismatches
    print('filtering out dgeoms:', too_close_mask.sum(), (~too_close_mask).sum())
    dgeom_info = dgeom_info[~too_close_mask]
    dgeoms = [td2.docking_geometry.DockingGeometry().from_dict(x)
              for _,x in dgeom_info.iterrows()]
    rep_dgeoms, rep_dgeom_indices = td2.docking_geometry.pick_docking_geometry_reps(
        'human', dgeoms, num_runs*4)
    assert len(rep_dgeoms) == num_runs*4
    return rep_dgeoms


all_rep_tcrs = {}
def get_rep_tcrs(mhc_class):
    'Save result since this takes a little time'
    global all_rep_tcrs
    if mhc_class not in all_rep_tcrs:
        tcrs = td2.sequtil.get_clean_and_nonredundant_ternary_tcrs_df(
            peptide_tcrdist_logical='and',
            verbose=True)
        all_rep_tcrs[mhc_class] = tcrs[tcrs.mhc_class==mhc_class].copy()
    return all_rep_tcrs[mhc_class]


def setup_for_design_redocking(
        mode,
        targets,
        outfile,
        scale_factor = 0.3, # magnitude of perturbations if mode == 'local'
        num_runs = 3,
        mhc_class = 1,
        pick_dgeoms_per_template = False, # set to True if targets have many diff peps
        # ^^^ only relevant for mode=='global'
):
    ''' targets should have columns:
    targetid (unique), chainseq, model_pdbfile, template_0_template_pdbfile,
    template_0_target_to_template_alignstring,

    template_0_template_pdbfile has to have an associated tdifile

    '''
    from numpy.linalg import norm
    assert targets.targetid.value_counts().max()==1
    assert mode in ['local','global']
    assert mhc_class == 1 # right now we are hardcoding chain order in a few places

    if mode=='local':
        dgeom_cols = 'torsion d tcr_unit_y tcr_unit_z mhc_unit_y mhc_unit_z'.split()
        # compute standard-devs for the 6 parameters:
        tcrs = get_rep_tcrs(mhc_class)
        param_sdevs = np.array([tcrs[col].std() for col in dgeom_cols])
        # perturbations are the same for all targets:
        perts = np.random.randn(num_runs*4, 6)
    else:
        if not pick_dgeoms_per_template:
            peptides = set(targets.chainseq.str.split('/').str.get(1))
            assert all(len(x)<20 for x in peptides)
            rep_dgeoms = get_rep_dgeoms_for_redocking(peptides, mhc_class, num_runs)
            assert len(rep_dgeoms) == num_runs*4


    dfl = []

    templates = sorted(set(targets.template_0_template_pdbfile))

    for tmp_pdbfile in templates:
        print('template:', tmp_pdbfile)

        tmp_pose = td2.pdblite.pose_from_pdb(tmp_pdbfile)
        _, _, tmp_nres_pmhc, _, tmp_nres = tmp_pose['chainbounds']
        transform_positions = list(range(tmp_nres_pmhc, tmp_nres))
        tdifile = tmp_pdbfile+'.tcrdock_info.json'
        with open(tdifile, 'r') as f:
            tdinfo = td2.tcrdock_info.TCRdockInfo().from_string(f.read())
        mhc_stub = td2.mhc_util.get_mhc_stub(tmp_pose, tdinfo=tdinfo)
        tcr_stub = td2.tcr_util.get_tcr_stub(tmp_pose, tdinfo)
        start_dgeom = td2.docking_geometry.DockingGeometry().from_stubs(
            mhc_stub, tcr_stub)
        if mode=='local':
            start_params = np.array([getattr(start_dgeom, col) for col in dgeom_cols])
            rep_dgeoms = []
            for pert in perts:
                new_params = start_params + scale_factor * pert * param_sdevs
                rep_dgeoms.append(td2.docking_geometry.DockingGeometry().from_dict(
                    {x:y for x,y in zip(dgeom_cols, new_params)}))
        elif pick_dgeoms_per_template:
            peptide = tmp_pose['chainseq'].split('/')[1]
            assert len(peptide) <20
            rep_dgeoms = get_rep_dgeoms_for_redocking([peptide], mhc_class, num_runs)
        assert len(rep_dgeoms) == num_runs*4
        dgeom_rmsds = td2.docking_geometry.compute_docking_geometries_distance_matrix(
            [start_dgeom], rep_dgeoms, 'human')
        print('template_dgeom_rmsds:', np.sort(dgeom_rmsds[0,:]),
              tmp_pdbfile.split('/')[-1])

        # confirm mhc stub is at origin
        dev1 = norm(mhc_stub['axes']-np.eye(3))
        dev2 = norm(mhc_stub['origin'])
        print('start mhc_stub:', norm(mhc_stub['axes']-np.eye(3)),
              norm(mhc_stub['origin']))
        assert dev1<1e-2 and dev2<1e-2

        my_targets = targets[targets.template_0_template_pdbfile==tmp_pdbfile]

        for _, l in my_targets.iterrows():
            alignstring = l.template_0_target_to_template_alignstring

            for run in range(num_runs):

                outl = dict(
                    targetid=f'{l.targetid}_{run}',
                    chainseq=l.chainseq,
                    design_targetid=l.targetid,
                    design_model_pdbfile=l.model_pdbfile,
                    min_dgeom_rmsd=np.min(dgeom_rmsds), # not specific to this run,
                    median_dgeom_rmsd=np.median(dgeom_rmsds), # a fxn of template...
                )

                for itmp in range(4):
                    dgeom_repno = itmp*num_runs + run # space out in hierarch tree
                    new_dgeom = rep_dgeoms[dgeom_repno]

                    new_tcr_stub = td2.docking_geometry.stub_from_docking_geometry(
                        new_dgeom)

                    # want a transform that maps tcr_stub to new_tcr_stub
                    R = new_tcr_stub['axes'].T @ tcr_stub['axes']
                    v = new_tcr_stub['origin'] - R@tcr_stub['origin']

                    transform = '{};{};{}'.format(
                        ','.join(str(x) for x in R.ravel()),
                        ','.join(str(x) for x in v),
                        ','.join(str(x) for x in transform_positions))

                    outl[f'template_{itmp}_template_pdbfile'] = tmp_pdbfile
                    outl[f'template_{itmp}_target_to_template_alignstring']= alignstring
                    outl[f'template_{itmp}_transform'] = transform
                dfl.append(outl)

    pd.DataFrame(dfl).to_csv(outfile, sep='\t', index=False)
    print('made:', outfile)
    return

setup_for_design_redocking(args.mode, targets, args.outfile, num_runs=args.num_runs)
