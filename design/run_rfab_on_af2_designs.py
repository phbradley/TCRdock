######################################################################################88


import argparse
parser = argparse.ArgumentParser(description="run rfab on alphafold dock designs")

parser.add_argument('--targets', required=True)
parser.add_argument('--outfile_prefix', required=True)
parser.add_argument('--dont_trust_model_dgeom', action='store_true',
                    help='hack for edge use case')

args = parser.parse_args()


# other imports
import design_paths
design_paths.setup_import_paths()
import tcrdock as td2
import pandas as pd
import numpy as np
from os.path import exists
from os import mkdir
import random
from collections import Counter
import itertools as it

import wrapper_tools
import design_stats
from tcrdock.docking_geometry import compute_docking_geometries_distance_matrix

targets = pd.read_table(args.targets)

required_cols = ('targetid organism mhc_class mhc peptide va ja cdr3a vb jb cdr3b '
                 'model_pdbfile').split()

for col in required_cols:
    assert col in targets.columns, f'Need {col} in --targets TSV file'


if targets.targetid.value_counts().max() > 1:
    print('WARNING: making targetid unique in', args.targets)
    targets['targetid'] = targets.targetid+'_'+targets.index.astype(str)


if not args.dont_trust_model_dgeom:
    ## add docking geometry info, compute distance to nearest template complexes
    dfl = []

    for _,l in targets.iterrows():
        outl = l.copy()
        dginfo = design_stats.compute_docking_geometry_info(l)
        for k,v in dginfo.items():
            outl[k] = v

        d_dgeom = td2.docking_geometry.DockingGeometry().from_dict(dginfo)

        # find likely templates, compute dgeom distance
        templates = pd.concat([td2.sequtil.ternary_info, td2.sequtil.new_ternary_info])
        t_dgeoms = [td2.docking_geometry.DockingGeometry().from_dict(x)
                    for _,x in templates.iterrows()]
        outl['min_td1']=td2.docking_geometry.compute_docking_geometries_distance_matrix(
            [d_dgeom], t_dgeoms, organism=l.organism).min()

        mask = ((templates.va == l.va) &(templates.ja == l.ja) &
                (templates.vb == l.vb) &(templates.jb == l.jb))
        if mask.sum():
            templates = templates[mask].copy()

            t_dgeoms = [td2.docking_geometry.DockingGeometry().from_dict(x)
                        for _,x in templates.iterrows()]
            outl['min_td2'] = compute_docking_geometries_distance_matrix(
                [d_dgeom], t_dgeoms, organism=l.organism).min()

        dfl.append(outl)

    targets = pd.DataFrame(dfl)

results = wrapper_tools.run_rf_antibody_on_designs(
    targets, args.outfile_prefix+'run_rfab'
)

if not args.dont_trust_model_dgeom:
    dfl = []
    for ii, row in results.reset_index(drop=True).iterrows():
        oldfile = row.old_model_pdbfile
        newfile = row.model_pdbfile

        cdr3_coords = []
        outl = row.copy()
        for fname, tag in [[oldfile,'old'], [newfile,'new']]:
            pose = td2.pdblite.pose_from_pdb(fname)
            assert pose['sequence'] == row.chainseq.replace('/','')
            cbs = [0]+list(it.accumulate(len(x) for x in row.chainseq.split('/')))
            pose = td2.pdblite.set_chainbounds_and_renumber(pose, cbs)

            tdinfo = design_stats.get_model_tdinfo(
                row.organism, row.mhc_class, row.mhc, row.chainseq,
                row.va, row.ja, row.cdr3a, row.vb, row.jb, row.cdr3b,
            )

            pose = td2.mhc_util.orient_pmhc_pose(pose, tdinfo=tdinfo)

            coords = []
            N, CA, C  = ' N  ', ' CA ', ' C  '

            for loop in [tdinfo.tcr_cdrs[3], tdinfo.tcr_cdrs[7]]:
                for pos in range(loop[0], loop[1]+1):
                    coords.append(pose['coords'][pose['resids'][pos]][N])
                    coords.append(pose['coords'][pose['resids'][pos]][CA])
                    coords.append(pose['coords'][pose['resids'][pos]][C])
            cdr3_coords.append(np.array(coords))

        natoms = cdr3_coords[0].shape[0]
        rmsd = np.sqrt(np.sum((cdr3_coords[0]-cdr3_coords[1])**2)/natoms)
        outl['cdr3_rmsd'] = rmsd
        dfl.append(outl)

    results = pd.DataFrame(dfl)
outfile = args.outfile_prefix+'rfab.tsv'
results.to_csv(outfile, sep='\t', index=False)
print('made:', outfile)

exit()

