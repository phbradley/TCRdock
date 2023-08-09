######################################################################################88
# rebuild 1000s of random loops into BOTH CDRs at the same time
# allow variable length CDRs
# vary the sequence between CAX and XF
# provide template info for CAXX and XXF
#


import argparse
parser = argparse.ArgumentParser(description="alphafold dock spectest")

parser.add_argument('--batch_num', type=int)
parser.add_argument('--num_batches', type=int)
parser.add_argument('--decoys_batch_num', type=int)
parser.add_argument('--decoys_num_batches', type=int)
parser.add_argument('--decoys', required=True)
parser.add_argument('--designs', required=True)
parser.add_argument('--outfile_prefix', required=True)
parser.add_argument('--num_recycle', type=int, default=3)
parser.add_argument('--model_name', default='model_2_ptm_ft')
parser.add_argument(
    '--model_params_file',
    default=('/home/pbradley/csdat/tcrpepmhc/amir/ft_params/'
             'model_2_ptm_ft_binder_20230729.pkl'))
parser.add_argument('--skip_ala_scan', action='store_true')
parser.add_argument('--skip_wt', action='store_true')
#parser.add_argument('--rundir')
#parser.add_argument('--loop_design_extra_args', default = '')
#parser.add_argument('--random_state',type=int, default=11)

args = parser.parse_args()


# other imports
import design_paths
design_paths.setup_import_paths()
import tcrdock as td2
import pandas as pd
from os.path import exists
from os import mkdir
import random
from collections import Counter

from design_stats import compute_simple_stats
from wrapper_tools import run_alphafold, run_mpnn

nterm_seq_stem = 3
cterm_seq_stem = 2


decoys = pd.read_table(args.decoys)
required_cols = 'organism mhc_class mhc peptide'.split()
for col in required_cols:
    assert col in decoys.columns, f'Need {col} in --decoys'

designs = pd.read_table(args.designs)
if 'targetid' not in designs.columns:
    designs['targetid'] = designs.index.astype(str)

target_cols = 'targetid organism mhc_class mhc peptide va ja cdr3a vb jb cdr3b'.split()
for col in target_cols:
    assert col in designs.columns, f'Need {col} in --designs'
designs = designs[target_cols]


if args.batch_num is None:
    args.batch_num = 0
    args.num_batches = 1

elif args.num_batches is None:
    args.num_batches = designs.shape[0]


if args.decoys_batch_num is None:
    args.decoys_batch_num = 0
    args.decoys_num_batches = 1

elif args.decoys_num_batches is None:
    args.decoys_num_batches = decoys.shape[0]

print(designs.shape[0], args.batch_num, args.num_batches)

dfl = []
for ii, ltcr in designs.iterrows():

    if ii%args.num_batches != args.batch_num:
        continue

    peptide = ltcr.peptide

    if not args.skip_ala_scan:
        #
        for jj, wtaa in enumerate(peptide):
            aa = 'A' if wtaa != 'A' else 'G'
            newpep = peptide[:jj] + aa + peptide[jj+1:]

            outl = ltcr.copy()
            outl['peptide'] = newpep
            outl['targetid'] = f'L{ii}_{ltcr.targetid}_mut_{jj}'

            dfl.append(outl)

    if not args.skip_wt:
        outl = ltcr.copy()
        outl['targetid'] = f'L{ii}_{ltcr.targetid}_wt'
        dfl.append(outl)

    for idecoy, lpmhc in decoys.iterrows():
        if lpmhc.peptide == ltcr.peptide:
            continue
        if idecoy%args.decoys_num_batches != args.decoys_batch_num:
            continue
        assert lpmhc.organism == ltcr.organism
        outl = ltcr.copy()
        outl['targetid'] = f'L{ii}_pep_{lpmhc.peptide}'
        outl['mhc_class'] = lpmhc.mhc_class
        outl['mhc'] = lpmhc.mhc
        outl['peptide'] = lpmhc.peptide
        dfl.append(outl)

tcrs = pd.DataFrame(dfl)

outdir = f'{args.outfile_prefix}_tmp/'
if not exists(outdir):
    mkdir(outdir)

td2.sequtil.setup_for_alphafold(
    tcrs, outdir, num_runs=1, use_opt_dgeoms=True, clobber=True,
)

targets = pd.read_table(outdir+'targets.tsv')
targets.rename(columns={'target_chainseq':'chainseq',
                        'templates_alignfile':'alignfile'}, inplace=True)
dfl = []
for l in targets.itertuples():
    seq = l.chainseq.replace('/','')
    posl = []
    for s in [l.cdr3a, l.cdr3b]:
        assert seq.count(s) == 1
        start = seq.index(s)
        posl.extend(range(start+nterm_seq_stem, start+len(s)-cterm_seq_stem))
    dfl.append(','.join([str(x) for x in posl]))
targets['designable_positions'] = dfl

# run alphafold
outprefix = f'{outdir}_afold'
targets = run_alphafold(
    targets, outprefix,
    num_recycle = args.num_recycle,
    model_name = args.model_name,
    model_params_file = args.model_params_file,
)


# compute stats
targets = compute_simple_stats(targets, extend_flex='barf')

# write results
targets.to_csv(f'{args.outfile_prefix}_final_results.tsv', sep='\t', index=False)

print('DONE')
