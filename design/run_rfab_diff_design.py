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

DIFFUSER_T = 50

import argparse
parser = argparse.ArgumentParser(description="run rfab diffusion designs")

parser.add_argument('--pmhc_pdbid', required=True)
parser.add_argument('--tcr_pdbid', required=True)
parser.add_argument('--num_designs', type=int, required=True)
parser.add_argument('--outfile_prefix', required=True)
parser.add_argument('--debug', action = 'store_true')
parser.add_argument('--design_other_cdrs', action = 'store_true')
parser.add_argument('--num_recycle', type=int, default=3)
parser.add_argument('--random_state', type=int)
parser.add_argument('--nterm_seq_stem', type=int, default=3,
                    help='number of non-designable positions at CDR3 N-terminus')
parser.add_argument('--cterm_seq_stem', type=int, default=2,
                    help='number of non-designable positions at CDR3 C-terminus')
parser.add_argument('--n_hotspot', type=int, default=3)
# parser.add_argument('--random_cdr3lens', action = 'store_true',
#                     help='choose cdr3lens from a big db of paired tcrs')
parser.add_argument('--cdr3_lens', help='specify cdr3a length range and cdr3b length '
                    'range, looks like "13-15,14-17" ')
parser.add_argument('--model_name', default='model_2_ptm_ft_binder',
                    help='this doesnt really matter but it has to start with '
                    '"model_2_ptm_ft"')

parser.add_argument('--model_params_file',
                    help='The default is a binder-fine-tuned model that was trained '
                    'on structures and a new distillation set')

args = parser.parse_args()


# other imports
import re
import design_paths
design_paths.setup_import_paths()
import tcrdock as td2
import pandas as pd
import numpy as np
from os.path import exists
from os import mkdir, system, popen
import random
from collections import Counter
import itertools as it
from copy import deepcopy
from timeit import default_timer as timer

import wrapper_tools
import design_stats

if args.model_params_file is None:
    args.model_params_file = design_paths.AF2_BINDER_FT_PARAMS

def get_my_designable_positions(tdinfo):
    ''' add _my in fxn name since get_designable_positions defined in design_stats.py

    this uses cmdline args:

    --design_other_cdrs
    --nterm_seq_stem
    --cterm_seq_stem

    and the CDR sequence positions defined in tdinfo

    '''
    global args # use cmdline flag info here

    designable_positions = []
    for ii, loop in enumerate(tdinfo.tcr_cdrs):
        if ii%4==2: # not designing cdr2.5
            continue
        elif ii%4==3: # cdr3
            npad, cpad = args.nterm_seq_stem, args.cterm_seq_stem
        else: # cdr1 or cdr2
            if not args.design_other_cdrs:
                continue
            npad, cpad = 0, 0

        designable_positions.extend(range(loop[0]+npad, loop[1]+1-cpad))
    return designable_positions

templates = pd.concat([td2.sequtil.ternary_info, td2.sequtil.new_ternary_info])

# temporary restrictions, could relax:
assert args.pmhc_pdbid in templates.index
assert args.tcr_pdbid in templates.index

# also temporary, makes rfabdiff output parsing easier below
assert templates.loc[args.pmhc_pdbid].mhc_class == 1


######################################################################################
## run rf antibody diffusion


tcr_pdbfile, design_loops = wrapper_tools.setup_rf_diff_tcr_template(
    args.tcr_pdbid,
    nterm_seq_stem=args.nterm_seq_stem,
    cterm_seq_stem=args.cterm_seq_stem,
)

pmhc_pdbfile, hotspot_string= wrapper_tools.setup_rf_diff_pmhc_template(
    args.pmhc_pdbid,
    n_hotspot=args.n_hotspot,
)


if args.cdr3_lens:
    pad = args.nterm_seq_stem + args.cterm_seq_stem # fixed cdr3 sequence from tcr tmplt
    a0,a1,b0,b1 = map(int, re.match('([0-9]+)-([0-9]+),([0-9]+)-([0-9]+)',
                                    args.cdr3_lens).groups())
    assert design_loops[2][:3] == 'H3:' and design_loops[5][:3] == 'L3:'
    assert a0>pad and b0>pad
    design_loops[2] = f'H3:{a0-pad}-{a1-pad}'
    design_loops[5] = f'L3:{b0-pad}-{b1-pad}'


if not args.design_other_cdrs:
    design_loops = [design_loops[2], design_loops[5]]

design_loopstring = '['+','.join(design_loops)+']'

rfabdiff_outprefix = args.outfile_prefix+'_rfabdiff'

# delete old pdbfiles, since we are parsing the logfile for looplen info (could fix)
for num in range(args.num_designs):
    pdbfile = f'{rfabdiff_outprefix}_{num}.pdb'
    if exists(pdbfile) and not args.debug:
        print('deleting old rf_diff pdbfile:', pdbfile)
        remove(pdbfile)


cmd = (f'{design_paths.RFDIFF_PYTHON} {design_paths.RFDIFF_SCRIPT} '
       f' --config-name antibody '
       f' inference.ckpt_override_path={design_paths.RFDIFF_CHK} '
       f' inference.num_designs={args.num_designs} '
       f' diffuser.T={DIFFUSER_T} '
       f' antibody.target_pdb={pmhc_pdbfile} '
       f' antibody.framework_pdb={tcr_pdbfile} '
       f' ppi.hotspot_res={hotspot_string} '
       f' antibody.design_loops={design_loopstring} '
       f' inference.output_prefix={rfabdiff_outprefix} '
       f' > {rfabdiff_outprefix}.log 2> {rfabdiff_outprefix}.err')

print(cmd)
start = timer()
if not args.debug:
    system(cmd)
rfdiff_time = timer()-start

######################################################################################
# now process the output and get setup to run mpnn
#
# make a targets dataframe with targetid, chainseq, model_pdbfile, and
#   designable_positions
#
dfl = []

tcr_row = templates.loc[args.tcr_pdbid]
tdifile = str(td2.util.path_to_db / tcr_row.pdbfile)+'.tcrdock_info.json'
with open(tdifile, 'r') as f:
    tdinfo_tcr_pdb = td2.tcrdock_info.TCRdockInfo().from_string(f.read())

pmhc_row = templates.loc[args.pmhc_pdbid]
tdifile = str(td2.util.path_to_db / pmhc_row.pdbfile)+'.tcrdock_info.json'
with open(tdifile, 'r') as f:
    tdinfo_pmhc_pdb = td2.tcrdock_info.TCRdockInfo().from_string(f.read())

#chainbounds come in handy
cbs_tcr_pdb  = [0] + list(it.accumulate(len(x) for x in  tcr_row.chainseq.split('/')))
cbs_pmhc_pdb = [0] + list(it.accumulate(len(x) for x in pmhc_row.chainseq.split('/')))

# figure out the actual loop lengths used -- this is fiddly!
logfile = rfabdiff_outprefix+'.log'
pattern = f'Timestep {DIFFUSER_T}, input to next step:'
lines = popen(f'grep -F "{pattern}" {logfile}').readlines()
assert len(lines) == args.num_designs, \
    f'rfab_diff failed or logfile parse err: {logfile}'
numloops = 6 if args.design_other_cdrs else 2
all_looplens = []
for line in lines:
    seq = line.split()[-1]
    looplens = [len(x) for x in re.findall('-+', seq)]
    assert len(looplens) == numloops
    print(looplens, seq)
    all_looplens.append(looplens)


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


    # shift things around to handle varying looplens
    looplens = all_looplens[num]
    ia, ib = (2, 5) if args.design_other_cdrs else (0, 1) # indices for cdr3s in looplens
    old_cdr3a_len = tdinfo_tcr_pdb.tcr_cdrs[3][1] - tdinfo_tcr_pdb.tcr_cdrs[3][0] +1
    new_cdr3a_len = looplens[ia] + args.nterm_seq_stem + args.cterm_seq_stem
    new_cdr3b_len = looplens[ib] + args.nterm_seq_stem + args.cterm_seq_stem

    ashift = nres_pmhc - cbs_tcr_pdb[2] # from tcr to pmhc
    bshift = ashift + new_cdr3a_len - old_cdr3a_len

    print(ashift, bshift, old_cdr3a_len, new_cdr3a_len, new_cdr3b_len, looplens)

    CORELEN = 13 ; assert len(tdinfo_tcr_pdb.tcr_core) == 2*CORELEN

    newcore = ([x+ashift for x in tdinfo_tcr_pdb.tcr_core[:CORELEN]]+
               [x+bshift for x in tdinfo_tcr_pdb.tcr_core[CORELEN:]])

    newcdrs = ([[x+ashift,y+ashift] for x,y in tdinfo_tcr_pdb.tcr_cdrs[:4]]+
               [[x+bshift,y+bshift] for x,y in tdinfo_tcr_pdb.tcr_cdrs[4:]])

    newcdrs[3][1] = newcdrs[3][0] + new_cdr3a_len-1
    newcdrs[7][1] = newcdrs[7][0] + new_cdr3b_len-1

    tdinfo = deepcopy(tdinfo_pmhc_pdb)
    tdinfo.tcr = tdinfo_tcr_pdb.tcr
    tdinfo.tcr_core = newcore
    tdinfo.tcr_cdrs = newcdrs

    tcr_coreseq = ''.join(pose['sequence'][x] for x in tdinfo.tcr_core)
    mhc_coreseq = ''.join(pose['sequence'][x] for x in tdinfo.mhc_core)
    print(tcr_coreseq)

    assert tcr_coreseq[ 1] == tcr_coreseq[14] == 'C'
    if args.nterm_seq_stem:
        assert tcr_coreseq[12] == tcr_coreseq[25] == 'C' # could fail on wonky tcr pdb
    else:
        assert tcr_coreseq[12] == tcr_coreseq[25] == 'G' # CDR3 will start with G

    dgeom = td2.docking_geometry.get_tcr_pmhc_docking_geometry(pose, tdinfo)

    peptide = pose['chainseq'].split('/')[1]
    assert peptide == tdinfo.pep_seq == pmhc_row.pep_seq

    atcr, btcr = tdinfo.tcr

    designable_positions = get_my_designable_positions(tdinfo)
    assert len(designable_positions) == sum(looplens)
    assert all(pose['sequence'][x] == 'G' for x in designable_positions)

    cdr3a_posl = range(tdinfo.tcr_cdrs[3][0], tdinfo.tcr_cdrs[3][1]+1)
    cdr3b_posl = range(tdinfo.tcr_cdrs[7][0], tdinfo.tcr_cdrs[7][1]+1)
    outl = dict(
        targetid = f'{args.pmhc_pdbid}_{args.tcr_pdbid}_{num}',
        chainseq = pose['chainseq'],
        model_pdbfile = outfile,
        designable_positions = ','.join(str(x) for x in designable_positions),
        pmhc_pdbid = args.pmhc_pdbid,
        tcr_pdbid = args.tcr_pdbid,
        organism = pmhc_row.organism,
        mhc_class = pmhc_row.mhc_class,
        mhc = pmhc_row.mhc_allele,
        va    = atcr[0],
        ja    = atcr[1],
        cdr3a_positions = ','.join(str(x) for x in cdr3a_posl),
        #cdr3a = atcr[2],
        vb    = btcr[0],
        jb    = btcr[1],
        cdr3b_positions = ','.join(str(x) for x in cdr3b_posl),
        #cdr3b = btcr[2],
        peptide = peptide,
    )
    for k,v in dgeom.to_dict().items():
        outl[k] = v
    dfl.append(outl)

targets = pd.DataFrame(dfl)


# run mpnn ############################################################################
outprefix = args.outfile_prefix+'_mpnn'
start = timer()
targets = wrapper_tools.run_mpnn(targets, outprefix, dry_run=args.debug)
mpnn_time = timer()-start

# update cdr3a, cdr3b sequence
cdr3s = []
for l in targets.itertuples():
    seq = l.chainseq.replace('/','')
    cdr3a = ''.join(seq[int(x)] for x in l.cdr3a_positions.split(','))
    cdr3b = ''.join(seq[int(x)] for x in l.cdr3b_positions.split(','))
    loopseq = ''.join(seq[int(x)] for x in l.designable_positions.split(','))
    cdr3s.append((cdr3a, cdr3b, loopseq))

targets['cdr3a'] = [x[0] for x in cdr3s]
targets['cdr3b'] = [x[1] for x in cdr3s]
targets['loopseq'] = [x[2] for x in cdr3s]

if args.nterm_seq_stem:
    assert all(targets.cdr3a.str.startswith('C'))
    assert all(targets.cdr3b.str.startswith('C'))

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
    use_new_templates=True,
)

af2_targets.rename(columns={'target_chainseq':'chainseq',
                            'templates_alignfile':'alignfile'}, inplace=True)

af2_targets['designable_positions'] = [
    ','.join(map(str, get_my_designable_positions(design_stats.get_row_tdinfo(x))))
    for x in af2_targets.itertuples()
]

# tricky-- we need to add in any design mutations in cdr1/cdr2 loops
# do this by updating the 'chainseq' column in af2_targets
#
if args.design_other_cdrs:
    for num, loopseq in enumerate(targets.loopseq):
        row = af2_targets.iloc[num]
        seq = list(row.chainseq.replace('/',''))
        for ii, pos in enumerate(map(int, row.designable_positions.split(','))):
            seq[pos] = loopseq[ii]
        seq = ''.join(seq) # it was a list
        cbs = [0]+list(it.accumulate(len(x) for x in row.chainseq.split('/')))
        new_cs = '/'.join(seq[a:b] for a,b in zip(cbs[:-1], cbs[1:]))
        af2_targets.loc[row.name, 'chainseq'] = new_cs

#af2_targets.to_csv('tmp.tsv', sep='\t', index=False)
#exit()

# run alphafold
outprefix = f'{outdir}afold'
start = timer()
af2_targets = wrapper_tools.run_alphafold(
    af2_targets, outprefix,
    num_recycle = args.num_recycle,
    model_name = args.model_name,
    model_params_file = args.model_params_file,
    ignore_identities = args.design_other_cdrs,
    dry_run = args.debug,
)
af2_time = timer()-start

# compute stats for alphafold comparison: tricky b/c sequences/lens wont match exactly
dgcols = ('d torsion mhc_unit_y mhc_unit_z mhc_unit_x_is_negative'
          ' tcr_unit_y tcr_unit_z tcr_unit_x_is_negative').split()
for col in dgcols:
    af2_targets[col] = list(targets[col])


# sanity check
assert all(np.array(targets.designable_positions.str.count(',')) ==
           np.array(af2_targets.designable_positions.str.count(',')))

# now we can add some stats-- this will compute dock rmsd to rfdiff model
# also fills in loop_seq, loop_seq2, peptide_loop_pae, etc
#
af2_targets = design_stats.compute_simple_stats(af2_targets)


######################################################################################
## now run rf_ab

start = timer()
rf2_targets = wrapper_tools.run_rf_antibody_on_designs(
    targets, args.outfile_prefix+'run_rfab',
    allow_pdb_chainseq_mismatch_for_tcr=True,# chainseq (ie mpnn seq) != rf-diff pdb seq
    dry_run = args.debug,
    delete_old_results = not args.debug,
)
rf2_time = timer()-start

######################################################################################
## create a final tsv file with info from the af2 and rf modeling:
copycols = ('model_pdbfile dgeom_rmsd chainseq cdr3a cdr3b '
            ' pmhc_tcr_pae loop_seq loop_seq2 '
            ' peptide_loop_pae peptide_plddt rfab_pmhc_tcr_pae rfab_pbind').split()
copycols += dgcols

for tag, results in [['af2',af2_targets],['rf2',rf2_targets]]:
    assert all(results.targetid==targets.targetid)
    for col in copycols:
        if col not in results.columns:
            print(f'{tag} results missing col: {col}')
        else:
            targets[tag+'_'+col] = results[col]

# compare the model structures:
start = timer()
all_models = {'rfdiff':targets, 'af2':af2_targets, 'rf2':rf2_targets}

for atag, amodels in all_models.items():
    for btag, bmodels in all_models.items():
        if btag <= atag:
            continue
        df = design_stats.compare_models(amodels, bmodels)
        for tag in ['cdr_rmsd','cdr3_rmsd','dgeom_rmsd']:
            targets[f'{atag}_{btag}_{tag}'] = list(df[tag])

targets['eval_time'] = (timer()-start)/args.num_designs

# compute dock rmsds between af and rf models
af2_dgeoms = [td2.docking_geometry.DockingGeometry().from_dict(x)
              for _,x in af2_targets.iterrows()]
rf2_dgeoms = [td2.docking_geometry.DockingGeometry().from_dict(x)
              for _,x in rf2_targets.iterrows()]
D = td2.docking_geometry.compute_docking_geometries_distance_matrix(
    af2_dgeoms, rf2_dgeoms)

targets['af2_rf2_dgeom_rmsd2'] = D[np.arange(args.num_designs),
                                   np.arange(args.num_designs)]

targets['rfdiff_time'] = rfdiff_time / args.num_designs
targets[  'mpnn_time'] =   mpnn_time / args.num_designs
targets[   'af2_time'] =    af2_time / args.num_designs
targets[   'rf2_time'] =    rf2_time / args.num_designs

outfile = args.outfile_prefix+'_final.tsv'
targets.to_csv(outfile, sep='\t', index=False)
print('made:', outfile)
print('DONE')
