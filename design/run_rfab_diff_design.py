######################################################################################88
#
# run diffusion (cdr3 only to start with)
# run mpnn
# run af2 (compare dock to design)
# run rf2 (compare dock to design and to af2 model)
#
# stats:
#
# - af2 pmhc_tcr_pae
# - af2 peptide_loop_pae
# - rf2 pae_interaction
# - rf2 p_bind
# - af2-rfdiff rmsd
# - rf2-rfdiff rmsd
# - rf2-af2 rmsd
#

ONLY_CDR3 = True
DIFFUSER_T = 50

import argparse
parser = argparse.ArgumentParser(description="run rfab diffusion designs")

parser.add_argument('--pmhc_pdbid', required=True)
parser.add_argument('--tcr_pdbid', required=True)
parser.add_argument('--num_designs', type=int, required=True)
parser.add_argument('--outfile_prefix', required=True)
parser.add_argument('--debug', action = 'store_true')
parser.add_argument('--num_recycle', type=int, default=3)
parser.add_argument('--random_state', type=int)
parser.add_argument('--model_name', default='model_2_ptm_ft_binder')
parser.add_argument(
    '--model_params_file',
    default=('/home/pbradley/csdat/tcrpepmhc/amir/ft_params/'
             'model_2_ptm_ft_binder_20230729.pkl')) # see read_amir_models

args = parser.parse_args()


# other imports
import design_paths
design_paths.setup_import_paths()
import tcrdock as td2
import pandas as pd
import numpy as np
from os.path import exists
from os import mkdir, system
import random
from collections import Counter
import itertools as it
from copy import deepcopy

import wrapper_tools
import design_stats


# temporary restrictions, could relax:
assert args.pmhc_pdbid in td2.sequtil.ternary_info.index
assert args.tcr_pdbid in td2.sequtil.ternary_info.index

# also temporary, makes rfabdiff output parsing easier below
assert td2.sequtil.ternary_info.loc[args.pmhc_pdbid].mhc_class == 1



######################################################################################
## run rf antibody diffusion
##
PY = '/home/pbradley/anaconda2/envs/SE3nv/bin/python'
EXE='/home/pbradley/gitrepos/rf_diffusion_netdbabsoft/rf_diffusion/run_inference.py'
CHK = ('/home/pbradley/gitrepos/rf_diffusion_netdbabsoft/rf_diffusion/'
       'model_weights/BFF_70.pt')

tcr_pdbfile, design_loops = wrapper_tools.setup_rf_diff_tcr_template(args.tcr_pdbid)
pmhc_pdbfile, hotspot_string= wrapper_tools.setup_rf_diff_pmhc_template(args.pmhc_pdbid)

if ONLY_CDR3:
    design_loops = [design_loops[2], design_loops[5]]

design_loopstring = '['+','.join(design_loops)+']'

rfabdiff_outprefix = args.outfile_prefix+'_rfabdiff'

cmd = (f'{PY} {EXE} --config-name antibody '
       f' inference.ckpt_override_path={CHK} '
       f' inference.num_designs={args.num_designs} '
       f' diffuser.T={DIFFUSER_T} '
       f' antibody.target_pdb={pmhc_pdbfile} '
       f' antibody.framework_pdb={tcr_pdbfile} '
       f' ppi.hotspot_res={hotspot_string} '
       f' antibody.design_loops={design_loopstring} '
       f' inference.output_prefix={rfabdiff_outprefix} '
       f' > {rfabdiff_outprefix}.log 2> {rfabdiff_outprefix}.err')

print(cmd)
if not args.debug:
    system(cmd)


######################################################################################
# now process the output and get setup to run mpnn
#
# make a targets dataframe with targetid, chainseq, model_pdbfile, and
#   designable_positions
#
dfl = []

tcr_row = td2.sequtil.ternary_info.loc[args.tcr_pdbid]
tdifile = str(td2.util.path_to_db / tcr_row.pdbfile)+'.tcrdock_info.json'
with open(tdifile, 'r') as f:
    tdinfo_tcr_pdb = td2.tcrdock_info.TCRdockInfo().from_string(f.read())

pmhc_row = td2.sequtil.ternary_info.loc[args.pmhc_pdbid]
tdifile = str(td2.util.path_to_db / pmhc_row.pdbfile)+'.tcrdock_info.json'
with open(tdifile, 'r') as f:
    tdinfo_pmhc_pdb = td2.tcrdock_info.TCRdockInfo().from_string(f.read())

#chainbounds come in handy
cbs_tcr_pdb  = [0] + list(it.accumulate(len(x) for x in  tcr_row.chainseq.split('/')))
cbs_pmhc_pdb = [0] + list(it.accumulate(len(x) for x in pmhc_row.chainseq.split('/')))


for num in range(args.num_designs):
    pdbfile = f'{rfabdiff_outprefix}_{num}.pdb'
    assert exists(pdbfile), f'rf_ab_diff failed for {pdbfile}'

    tcr_pose = td2.pdblite.pose_from_pdb(pdbfile)
    assert len(tcr_pose['chains']) == 3 # H L T
    pose = deepcopy(tcr_pose)

    tcr_pose = td2.pdblite.delete_chains(tcr_pose, [2])
    pose = td2.pdblite.delete_chains(pose, [0,1])
    pose = td2.pdblite.append_chains(pose, tcr_pose, [0,1])

    cbs = pose['chainbounds'][:]
    peplen = len(pmhc_row.pep_seq)
    nres_pmhc = cbs[1]
    nres_mhc = nres_pmhc - peplen
    assert pmhc_row.mhc_class == 1
    cbs = [0, nres_mhc, nres_pmhc] + cbs[-2:] ## assumes mhc_class =1 !
    pose = td2.pdblite.set_chainbounds_and_renumber(pose, cbs)

    assert len(pose['chains']) == 4 # mhc peptide tcra tcrb
    outfile = pdbfile+'_renumber.pdb'
    td2.pdblite.dump_pdb(pose, outfile)

    assert [nres_mhc, nres_pmhc] == cbs_pmhc_pdb[1:3]

    shift = nres_pmhc - cbs_tcr_pdb[2] # from tcr to pmhc


    tdinfo = deepcopy(tdinfo_pmhc_pdb)
    tdinfo.tcr = tdinfo_tcr_pdb.tcr
    tdinfo.tcr_core = [x+shift for x in tdinfo_tcr_pdb.tcr_core]
    tdinfo.tcr_cdrs = [[x+shift,y+shift] for x,y in tdinfo_tcr_pdb.tcr_cdrs]

    tcr_coreseq = ''.join(pose['sequence'][x] for x in tdinfo.tcr_core)
    mhc_coreseq = ''.join(pose['sequence'][x] for x in tdinfo.mhc_core)
    #print('mod_tcr_coreseq:', tcr_coreseq)
    #print('mod_mhc_coreseq:', mhc_coreseq)

    assert tcr_coreseq[ 1] == tcr_coreseq[14] == 'C'
    assert tcr_coreseq[12] == tcr_coreseq[25] == 'G' # assumes we are dfdiffing cdr3

    dgeom = td2.docking_geometry.get_tcr_pmhc_docking_geometry(pose, tdinfo)

    peptide = pose['chainseq'].split('/')[1]
    assert peptide == tdinfo.pep_seq == pmhc_row.pep_seq

    atcr, btcr = tdinfo.tcr

    designable_positions = (
        list(range(tdinfo.tcr_cdrs[3][0], tdinfo.tcr_cdrs[3][1]+1))+
        list(range(tdinfo.tcr_cdrs[7][0], tdinfo.tcr_cdrs[7][1]+1))
    )

    assert all(pose['sequence'][x] == 'G' for x in designable_positions)

    outl = dict(
        targetid = f'{args.pmhc_pdbid}_{args.tcr_pdbid}_{num}',
        chainseq = pose['chainseq'],
        model_pdbfile = outfile,
        designable_positions = ','.join(str(x) for x in designable_positions),
        pmhc_pdbid = args.pmhc_pdbid,
        tcr_pdbid = args.tcr_pdbid,
        only_cdr3 = ONLY_CDR3,
        organism = pmhc_row.organism,
        mhc_class = pmhc_row.mhc_class,
        mhc = pmhc_row.mhc_allele,
        va    = atcr[0],
        ja    = atcr[1],
        #cdr3a = atcr[2],
        vb    = btcr[0],
        jb    = btcr[1],
        #cdr3b = btcr[2],
        peptide = peptide,
    )
    for k,v in dgeom.to_dict().items():
        outl[k] = v
    dfl.append(outl)

targets = pd.DataFrame(dfl)

print(targets.iloc[0])

# run mpnn ############################################################################
outprefix = args.outfile_prefix+'_mpnn'
targets = wrapper_tools.run_mpnn(targets, outprefix, dry_run=args.debug)

# update cdr3a, cdr3b sequence
cdr3s = []
for l in targets.itertuples():
    seq = l.chainseq.replace('/','')
    posl = [int(x) for x in l.designable_positions.split(',')]
    loopseq = ''
    for i,pos in enumerate(posl):
        if i and pos != posl[i-1]+1:
            loopseq += ','
        loopseq += seq[pos]
    assert loopseq.count(',') == 1
    cdr3s.append(loopseq.split(','))

targets['cdr3a'] = [x[0] for x in cdr3s]
targets['cdr3b'] = [x[1] for x in cdr3s]

print(targets.iloc[0])

######################################################################################
# now run alphafold
outdir = f'{args.outfile_prefix}_afold/'
if not exists(outdir):
    mkdir(outdir)
cols = ('organism mhc mhc_class peptide va ja cdr3a vb jb cdr3b tcr_pdbid pmhc_pdbid'
        ' targetid').split()
af2_targets = targets[cols]
af2_targets = td2.sequtil.setup_for_alphafold(
    af2_targets, outdir, num_runs=1, use_opt_dgeoms=True, clobber=True,
    force_tcr_pdbids_column='tcr_pdbid',
    force_pmhc_pdbids_column='pmhc_pdbid',
)

af2_targets.rename(columns={'target_chainseq':'chainseq',
                            'templates_alignfile':'alignfile'}, inplace=True)

# run alphafold
outprefix = f'{outdir}afold'
af2_targets = wrapper_tools.run_alphafold(
    af2_targets, outprefix,
    num_recycle = args.num_recycle,
    model_name = args.model_name,
    model_params_file = args.model_params_file,
    dry_run = args.debug,
)

# compute stats for alphafold comparison: tricky b/c sequences wont match exactly
dgcols = ('d torsion mhc_unit_y mhc_unit_z mhc_unit_x_is_negative'
          ' tcr_unit_y tcr_unit_z tcr_unit_x_is_negative').split()
for col in dgcols:
    af2_targets[col] = list(targets[col])
dfl = []
for _,row in af2_targets.iterrows():
    seq = row.chainseq.replace('/','')
    posl = []
    for cdr in [row.cdr3a, row.cdr3b]:
        assert seq.count(cdr)==1
        posl.extend([seq.index(cdr)+x for x in range(len(cdr))])
    outl = row.copy()
    outl['designable_positions'] = ','.join(str(x) for x in posl)
    dfl.append(outl)
af2_targets = pd.DataFrame(dfl)

# now we can add some stats
af2_targets = design_stats.compute_simple_stats(af2_targets)


######################################################################################
## now run rf_ab

rf2_targets = wrapper_tools.run_rf_antibody_on_designs(
    targets, args.outfile_prefix+'run_rfab',
    allow_pdb_chainseq_mismatch_for_tcr=True,
    dry_run = args.debug,
    delete_old_results = True,
)

print(rf2_targets.iloc[0])


## create a final tsv file
copycols = ('model_pdbfile dgeom_rmsd chainseq cdr3a cdr3b pmhc_tcr_pae'
            ' peptide_loop_pae peptide_plddt rfab_pmhc_tcr_pae rfab_pbind').split()
copycols += dgcols

for tag, results in [['af2',af2_targets],['rf2',rf2_targets]]:
    assert all(results.targetid==targets.targetid)
    for col in copycols:
        if col not in results.columns:
            print(f'{tag} results missing col: {col}')
        else:
            targets[tag+'_'+col] = results[col]
outfile = args.outfile_prefix+'_final.tsv'
targets.to_csv(outfile, sep='\t', index=False)
print('made:', outfile)
