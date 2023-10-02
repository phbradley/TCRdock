######################################################################################88

import argparse
parser = argparse.ArgumentParser(
    description="run peptide specificity testing on good designs",
    epilog=
'''
This script is for evaluating a small-ish number of good designs
for binding to a set of decoy peptides.

A TSV file of designs is passed with the --designs <filename> option.
The required fields are:
   organism mhc_class mhc peptide va ja cdr3a vb jb cdr3b
   chainseq (necessary if non-cdr3 loops were designed)
   targetid (an integer index will be added if the targetid column is missing)


A TSV file of decoy peptides is passed with the --decoys <filename> option.
The required fields are:
   organism mhc_class mhc peptide

You don't need the --decoys file if you use either of the next two options:
--ala_scan will add all the single-ala mutant peptides as decoys
--x_scan will add all the single-aa mutant peptides as decoys

Example command line:

python dock_spectest.py --designs my_good_designs.tsv \
    --decoys my_decoy_peptides.tsv  --outfile_prefix test1

OR (we can skip --decoys if --ala_scan or --x_scan are present)

python dock_spectest.py --designs my_good_designs.tsv \
    --ala_scan  --outfile_prefix test2

# this will divide the work into 5 batches and run batch 0
python dock_spectest.py --designs my_good_designs.tsv \
    --x_scan  --outfile_prefix test3_b0 \
    --num_batches 5 --batch_num 0


email pbradley@fredhutch.org with questions

''',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument('--decoys')
parser.add_argument('--designs', required=True)
parser.add_argument('--outfile_prefix', required=True)
parser.add_argument('--batch_num', type=int)
parser.add_argument('--num_batches', type=int)
parser.add_argument('--num_recycle', type=int, default=3)
parser.add_argument('--force_pmhc_pdbids_column',
                    help='For rfdiff designs, when we remodel with alphafold '
                    'we force the pmhc and tcr templates to match the rfdiff '
                    'templates. Use "--force_pmhc_pdbids_column pmhc_pdbid" to '
                    'use the same templates here (assuming pmhc_pdbid is a column '
                    'in the --designs TSVfile')
parser.add_argument('--force_tcr_pdbids_column',
                    help='see --force_pmhc_pdbids_column ')
parser.add_argument('--model_name', default='model_2_ptm_ft_binder',
                    help='this doesnt really matter but it has to start with '
                    '"model_2_ptm_ft"')

parser.add_argument('--model_params_file',
                    help='The default is a binder-fine-tuned model that was trained '
                    'on structures and a new distillation set')

parser.add_argument('--ala_scan', action='store_true',
                    help='For each design, add additional decoy pmhcs corresponding to '
                    'single Ala (or Gly) peptide mutants')

parser.add_argument('--x_scan', action='store_true',
                    help='For each design, add additional decoy pmhcs corresponding to '
                    'all single AA peptide mutants')

parser.add_argument('--other_scan', action='store_true',
                    help='For each design, add additional decoy pmhcs from the same '
                    'or other mhcs and lens')

parser.add_argument('--skip_wt', action='store_true',
                    help='Otherwise the wt pmhc will be included in the calcs')

parser.add_argument('--only_cdr3_design', action='store_true',
                    help='use this flag to signal that all the sequence info is '
                    'present in the v genes and cdr3 sequences')

#parser.add_argument('--rundir')
#parser.add_argument('--loop_design_extra_args', default = '')
#parser.add_argument('--random_state',type=int, default=11)

args = parser.parse_args()

# other imports
import copy
import design_paths
design_paths.setup_import_paths()
import tcrdock as td2
import numpy as np
import pandas as pd
from os.path import exists
import os
import random
from collections import Counter
from tcrdock.tcrdist.amino_acids import amino_acids
import tcrdock.util
from design_stats import compute_simple_stats
from wrapper_tools import run_alphafold, run_mpnn

if args.model_params_file is None:
    args.model_params_file = design_paths.AF2_BINDER_FT_PARAMS

# these are really not very important, just for calc peptide_loop_pae in
#   design_stats.compute_simple_stats
# and it's not quite right anyhow if we are designing other loops...
#
nterm_seq_stem = 3
cterm_seq_stem = 2

if args.other_scan:
    other_fname = tcrdock.util.path_to_db / 'hla_binding_decoy_peptides_v1.tsv'
    other_decoys = pd.read_table(other_fname)

if args.decoys is None:
    assert args.ala_scan or args.x_scan or args.other_scan
    decoys = pd.DataFrame() # empty
else:
    decoys = pd.read_table(args.decoys)
    required_cols = 'organism mhc_class mhc peptide'.split()
    for col in required_cols:
        assert col in decoys.columns, f'Need {col} in --decoys'


designs = pd.read_table(args.designs)
if 'targetid' not in designs.columns:
    designs['targetid'] = designs.index.astype(str)

elif designs.targetid.value_counts().max()>1: # make targetid unique if necessary
    designs['targetid'] = designs.targetid + "_" + designs.index.astype(str)


# here we only need chainseq if there are non-cdr3 mutations
required_cols = ('targetid organism mhc_class mhc peptide va ja cdr3a '
                 'vb jb cdr3b'.split())
if not args.only_cdr3_design:
    required_cols.append('chainseq')

for col in required_cols:
    assert col in designs.columns, f'Need {col} in --designs'
#designs = designs[target_cols]


if args.batch_num is None:
    # do everything in one go
    args.batch_num = 0
    args.num_batches = 1
else:
    # if you pass --batch_num you need to pass num_batches also
    assert args.num_batches is not None
    assert 0 <= args.batch_num < args.num_batches

dfl = []
for _, ltcr in designs.iterrows():

    wt_peptide = ltcr.peptide

    for ip, lpmhc in decoys.iterrows(): # decoys may be empty...
        assert lpmhc.organism == ltcr.organism
        outl = ltcr.copy()
        outl['targetid'] = f'{ltcr.targetid}_pep_{ip}_{lpmhc.peptide}'
        outl['mhc_class'] = lpmhc.mhc_class
        outl['mhc'] = lpmhc.mhc
        outl['peptide'] = lpmhc.peptide
        dfl.append(outl)


    if not args.skip_wt:
        outl = ltcr.copy()
        outl['targetid'] = f'{ltcr.targetid}_wt'
        dfl.append(outl)


    if args.ala_scan:
        for jj, wtaa in enumerate(wt_peptide):
            aa = 'A' if wtaa != 'A' else 'G'
            newpep = wt_peptide[:jj] + aa + wt_peptide[jj+1:]

            outl = ltcr.copy()
            outl['peptide'] = newpep
            outl['targetid'] = f'{ltcr.targetid}_alascan_{jj}'

            dfl.append(outl)

    if args.x_scan:
        for jj, wtaa in enumerate(wt_peptide):
            for aa in amino_acids:
                if aa == wtaa:
                    continue
                newpep = wt_peptide[:jj] + aa + wt_peptide[jj+1:]

                outl = ltcr.copy()
                outl['peptide'] = newpep
                outl['targetid'] = f'{ltcr.targetid}_xscan_{jj}{aa}'

                dfl.append(outl)

    if args.other_scan:
        for mhc in other_decoys.mhc.unique():
            if mhc == ltcr.mhc:
                num = 100
            else:
                num = 25
            odf = other_decoys[other_decoys.mhc==mhc].sample(num, random_state=10)
            for _,lpmhc in odf.iterrows():
                assert lpmhc.organism == ltcr.organism
                outl = ltcr.copy()
                mhc_tag = mhc.replace('*','').replace(':','')
                outl['targetid'] = f'{ltcr.targetid}_pep_{mhc_tag}_{lpmhc.peptide}'
                outl['mhc_class'] = lpmhc.mhc_class
                outl['mhc'] = lpmhc.mhc
                outl['peptide'] = lpmhc.peptide
                dfl.append(outl)

tcrs = pd.DataFrame(dfl)
if args.num_batches:
    mask = np.arange(tcrs.shape[0])%args.num_batches == args.batch_num
    print(f'batch {args.batch_num}:: subset to {mask.sum()} out of {tcrs.shape[0]}',
          'total jobs')
    tcrs = tcrs[mask].reset_index(drop=True)


outdir = f'{args.outfile_prefix}_tmp/'
os.makedirs(outdir, exist_ok=True)

# rename these old columns if they exist, before they get overwritten by af2 setup
tcrs.rename(
    columns={'chainseq':'old_chainseq', 'alignfile':'old_alignfile'}, inplace=True)

targets = td2.sequtil.setup_for_alphafold(
    tcrs, outdir, num_runs=1,
    use_opt_dgeoms=True, clobber=True,
    force_pmhc_pdbids_column=args.force_pmhc_pdbids_column,
    force_tcr_pdbids_column=args.force_tcr_pdbids_column,
    use_new_templates = True,
)

# fixup legacy column names...
targets.rename(columns={'target_chainseq':'chainseq',
                        'templates_alignfile':'alignfile'}, inplace=True)
dfl = []
for _, l in targets.iterrows():
    outl = l.copy()
    if not args.only_cdr3_design:
        # tricky-- we have to make any additional non-cdr3 mutations
        old_cs = l.old_chainseq.split('/')
        new_cs = l.chainseq.split('/')
        assert len(old_cs) == len(new_cs) == l.mhc_class + 3

        mutations = [0,0]
        for ich in range(2):
            old_seq = old_cs[-2+ich]
            new_seq = new_cs[-2+ich]

            al = td2.sequtil.blosum_align(new_seq, old_seq)
            new_seq = list(new_seq)
            for i,j in al.items():
                if new_seq[i] != old_seq[j]:
                    #print(f'mutation from {new_seq[i]} to {old_seq[j]} at {i},{j}')
                    new_seq[i] = old_seq[j]
                    mutations[ich] += 1
            new_cs[-2+ich] = ''.join(new_seq)

        outl['chainseq'] = '/'.join(new_cs)
        assert len(outl.chainseq) == len(l.chainseq)
        print(f'tcra_muts: {mutations[0]:2d} tcrb_muts: {mutations[1]:2d} {l.targetid}')

    # setup the designable positions to mark the cdr3a and cdr3b loops, for
    #  peptide_loop_pae in
    seq = l.chainseq.replace('/','')
    posl = []
    for s in [l.cdr3a, l.cdr3b]:
        assert seq.count(s) == 1
        start = seq.index(s)
        posl.extend(range(start+nterm_seq_stem, start+len(s)-cterm_seq_stem))
    outl['designable_positions'] = ','.join([str(x) for x in posl])
    dfl.append(outl)

targets = pd.DataFrame(dfl)

# run alphafold
outprefix = f'{outdir}_afold'
targets = run_alphafold(
    targets, outprefix,
    num_recycle = args.num_recycle,
    model_name = args.model_name,
    model_params_file = args.model_params_file,
    ignore_identities=True, # since we may have mutated chainseq
)

# compute stats
targets = compute_simple_stats(targets, extend_flex='barf')

# write results
targets.to_csv(f'{args.outfile_prefix}_final_results.tsv', sep='\t', index=False)

print('DONE')
