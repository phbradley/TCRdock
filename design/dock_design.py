######################################################################################88
# rebuild 1000s of random loops into BOTH CDRs at the same time
# allow variable length CDRs
# vary the sequence between CAX and XF
# provide template info for CAXX and XXF
#


import argparse
parser = argparse.ArgumentParser(description="alphafold dock design")

parser.add_argument('--num_designs', type=int, required=True)
parser.add_argument('--pmhc_targets', required=True)
parser.add_argument('--outfile_prefix', required=True)
parser.add_argument('--allow_mhc_mismatch', action='store_true')
parser.add_argument('--num_recycle', type=int, default=3)
parser.add_argument('--random_state', type=int)
parser.add_argument('--model_name', default='model_2_ptm_ft')
parser.add_argument(
    '--model_params_file',
    default=('/home/pbradley/csdat/tcrpepmhc/amir/ft_params/'
             'model_2_ptm_ft_binder_20230729.pkl'))
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


## hard-coded
nterm_seq_stem = 3
cterm_seq_stem = 2
#nterm_align_stem = 4
#cterm_align_stem = 3
force_native_seq_stems = True # since these aren't being designed...
## defaults ##########



pmhc_targets = pd.read_table(args.pmhc_targets)
required_cols = 'organism mhc_class mhc peptide'.split()
for col in required_cols:
    assert col in pmhc_targets.columns, f'Need {col} in --pmhc_targets'


tcrs_file = '/home/pbradley/csdat/big_covid/big_combo_tcrs_2022-01-22.tsv'
print('reading:', tcrs_file)
big_tcrs_df = pd.read_table(tcrs_file)

# exclude extreme len cdr3s
badmask = ((big_tcrs_df.cdr3a.str.len()<9)|
           (big_tcrs_df.cdr3b.str.len()<9)|
           (big_tcrs_df.cdr3a.str.len()>17)|
           (big_tcrs_df.cdr3b.str.len()>17))
big_tcrs_df = big_tcrs_df[~badmask]

targets_dfl = []

pmhcs = pmhc_targets.sample(n=args.num_designs, replace=True,
                            random_state=args.random_state)
cdr3s = big_tcrs_df.sample(n=args.num_designs, replace=True,
                           random_state=args.random_state)

dfl = []
for (_,lpmhc), lcdr3 in zip(pmhcs.iterrows(), cdr3s.itertuples()):

    # look for tcrs with the same allele
    templates = td2.sequtil.ternary_info.copy()
    if not args.allow_mhc_mismatch:
        templates = templates[(templates.mhc_class==lpmhc.mhc_class)&
                              (templates.mhc_allele==lpmhc.mhc)]
        if templates.shape[0] == 0:
            print('no matching templates found for mhc:',
                  lpmhc.mhc)
            exit(1)

    outl = lpmhc.copy()
    ltcr = templates.sample(n=1).iloc[0]
    cdr3a = lcdr3.cdr3a
    cdr3b = lcdr3.cdr3b
    # preserve 1st 3 and last 2 rsds from template tcr (va,ja,vb,jb)
    cdr3a = (ltcr.cdr3a[:nterm_seq_stem] +
             cdr3a[nterm_seq_stem:-cterm_seq_stem] +
             ltcr.cdr3a[-cterm_seq_stem:])
    cdr3b = (ltcr.cdr3b[:nterm_seq_stem] +
             cdr3b[nterm_seq_stem:-cterm_seq_stem] +
             ltcr.cdr3b[-cterm_seq_stem:])
    outl['va'] = ltcr.va
    outl['ja'] = ltcr.ja
    outl['cdr3a'] = cdr3a
    outl['vb'] = ltcr.vb
    outl['jb'] = ltcr.jb
    outl['cdr3b'] = cdr3b
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
outprefix = f'{outdir}_afold1'
targets = run_alphafold(
    targets, outprefix,
    num_recycle = args.num_recycle,
    model_name = args.model_name,
    model_params_file = args.model_params_file,
    #dry_run = True,
)

# run mpnn
outprefix = f'{outdir}_mpnn'
targets = run_mpnn(targets, outprefix, extend_flex='barf')

# run alphafold again
outprefix = f'{outdir}_afold2'
targets = run_alphafold(
    targets, outprefix,
    num_recycle=args.num_recycle,
    model_name = args.model_name,
    model_params_file = args.model_params_file,
    ignore_identities = True,
)

# compute stats
targets = compute_simple_stats(targets, extend_flex='barf')

# write results
targets.to_csv(f'{args.outfile_prefix}_final_results.tsv', sep='\t', index=False)

print('DONE')
