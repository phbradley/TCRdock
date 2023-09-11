
import argparse
parser = argparse.ArgumentParser(
    description="alphafold dock design",
    epilog='''
Flexible dock design using the "random loops" approach

input is a TSV file with peptide-MHC target info (allele name, peptide seq)

The script repeats the following steps '--num_designs' times:

STEP 1. pick a random pmhc target from list

STEP 2. pick a template tcr for that pmhc target. By default this will be a tcr that
binds to the same MHC allele. This template contributes the following information:
* va, ja, vb, jb (the V and J genes for the alpha and beta chains)
* the first 3 and last 2 residues of the CDR3 loops

    --allow_mhc_mismatch will expand the set of potential templates to include all
tcrs that bind the same MHC class

STEP 3. pick CDR3 loops from a random paired TCR. Mutate the first 3 and last 2 aas
to match the template tcr from STEP 2.

STEP 4. Provide this information (peptide,MHC,V/J genes, CDR3 sequences) to a
modified alphafold TCR docking protocol

STEP 5. Re-design the CDR3 loops (excluding first 3 and last 2 residues) using MPNN

STEP 6. Re-dock the TCR to the pMHC using the same alphafold docking protocol used
in step 4.

STEP 7. Compute final stats like pmhc_tcr_pae, peptide_loop_pae, and dock-rmsd between
first and second alphafold models.



Example command line:

python dock_design.py --pmhc_targets my_pmhc_targets.tsv \\
    --num_designs 10  --outfile_prefix dock_design_test1

The --pmhc_targets file should have these columns:
    * organism ('human' or 'mouse')
    * mhc_class (1 or 2)
    * mhc (e.g. "A*02:01")
    * peptide (e.g. "GILGFVFTL")

$ head my_pmhc_targets.tsv
organism	mhc_class	mhc	peptide
human	1	A*01:01	EVDPIGHLY

email pbradley@fredhutch.org with questions

''',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument('--num_designs', type=int, required=True)
parser.add_argument('--pmhc_targets', required=True)
parser.add_argument('--outfile_prefix', required=True)
parser.add_argument('--allow_mhc_mismatch', action='store_true')
parser.add_argument('--design_other_cdrs', action='store_true')
parser.add_argument('--num_recycle', type=int, default=3)
parser.add_argument('--random_state', type=int)
parser.add_argument('--model_name', default='model_2_ptm_ft_binder')
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
from timeit import default_timer as timer

import design_stats
from wrapper_tools import run_alphafold, run_mpnn


## hard-coded -- these control how much sequence is retained from tcr template cdr3s
nterm_seq_stem = 3
cterm_seq_stem = 2
## defaults ##########


# read the targets
pmhc_targets = pd.read_table(args.pmhc_targets)
required_cols = 'organism mhc_class mhc peptide'.split()
for col in required_cols:
    assert col in pmhc_targets.columns, f'Need {col} in --pmhc_targets'

# read the big paired tcr database, this provides the random cdr3a/cdr3b pairs
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

# sample --num_designs pmhcs and cdr3a/b pairs
pmhcs = pmhc_targets.sample(n=args.num_designs, replace=True,
                            random_state=args.random_state)

cdr3s = big_tcrs_df.sample(n=args.num_designs, replace=True,
                           random_state=args.random_state)

dfl = []
for (_,lpmhc), lcdr3 in zip(pmhcs.iterrows(), cdr3s.itertuples()):

    # look for tcrs with the same allele
    templates = td2.sequtil.ternary_info.copy()
    templates = templates[templates.organism == lpmhc.organism].copy()
    if args.allow_mhc_mismatch: # only enforce same mhc_class
        templates = templates[templates.mhc_class==lpmhc.mhc_class]
    else: # require same mhc class and same mhc allele
        templates = templates[(templates.mhc_class==lpmhc.mhc_class)&
                              (templates.mhc_allele==lpmhc.mhc)]
    if templates.shape[0] == 0:
        print('ERROR no matching templates found for mhc:',
              lpmhc.mhc)
        exit(1)

    outl = lpmhc.copy()
    ltcr = templates.sample(n=1).iloc[0]
    cdr3a = lcdr3.cdr3a
    cdr3b = lcdr3.cdr3b
    # preserve 1st 3 and last 2 cdr3 rsds from template tcr (set by n/cterm_seq_stem)
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
    outl['tcr_template_pdbid'] = ltcr.pdbid
    dfl.append(outl)

tcrs = pd.DataFrame(dfl)

outdir = f'{args.outfile_prefix}_tmp/'
if not exists(outdir):
    mkdir(outdir)

targets = td2.sequtil.setup_for_alphafold(
    tcrs, outdir, num_runs=1, use_opt_dgeoms=True, clobber=True,
    force_tcr_pdbids_column='tcr_template_pdbid',
)

targets.rename(columns={'target_chainseq':'chainseq',
                        'templates_alignfile':'alignfile'}, inplace=True)
dfl = []
for l in targets.itertuples():
    posl = []
    if args.design_other_cdrs: # designing the other cdr loops here
        tdinfo = design_stats.get_row_tdinfo(l)
        for ii, loop in enumerate(tdinfo.tcr_cdrs):
            if ii in [2,6]: # skip cdr2.5
                continue
            npad, cpad = (nterm_seq_stem, cterm_seq_stem) if ii in [3,7] else \
                         (0,0)
            posl.extend(range(loop[0]+npad, loop[1]+1-cpad))
    else:
        seq = l.chainseq.replace('/','')
        for s in [l.cdr3a, l.cdr3b]:
            assert seq.count(s) == 1
            start = seq.index(s)
            posl.extend(range(start+nterm_seq_stem, start+len(s)-cterm_seq_stem))
    dfl.append(','.join([str(x) for x in posl]))
targets['designable_positions'] = dfl


# run alphafold
outprefix = f'{outdir}_afold1'
start = timer()
targets = run_alphafold(
    targets, outprefix,
    num_recycle = args.num_recycle,
    model_name = args.model_name,
    model_params_file = args.model_params_file,
    #dry_run = True,
)
af2_time = timer()-start

# compute stats; most will be over-written but this saves docking geometry info
# so at the end we will get an rmsd between the mpnn-input pose and the final
# alphafold re-docked pose
targets = design_stats.compute_simple_stats(targets, extend_flex='barf')

# run mpnn
outprefix = f'{outdir}_mpnn'
start = timer()
targets = run_mpnn(targets, outprefix, extend_flex='barf')
mpnn_time = timer()-start

# run alphafold again
outprefix = f'{outdir}_afold2'
start = timer()
targets = run_alphafold(
    targets, outprefix,
    num_recycle=args.num_recycle,
    model_name = args.model_name,
    model_params_file = args.model_params_file,
    ignore_identities = True, # since mpnn changed sequence...
)
af2_time += timer()-start

# compute stats again. this should compute docking rmsd to the original mpnn dock
targets = design_stats.compute_simple_stats(targets, extend_flex='barf')

# write results
targets['af2_time'] = af2_time/args.num_designs
targets['mpnn_time'] = mpnn_time/args.num_designs

targets.to_csv(f'{args.outfile_prefix}_final_results.tsv', sep='\t', index=False)

print('DONE')
