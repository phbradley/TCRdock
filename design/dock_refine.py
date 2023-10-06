import argparse

parser = argparse.ArgumentParser(
    description="iterative dock design refinement",
epilog='''

Iterative refinement of a starting pool of flexible-dock TCR designs
distributed across multiple processes.

The criterion for defining an improved design is given by the --sort_tag flag.

Current criteria (lower is better):
* pmhc_tcr_pae: AF2 pmc_tcr_pae
* combo_score: af2_pmhc_tcr_pae + rfab_pmhc_tcr_pae + af2_rfab_cdr3_rmsd
* combo_score_wtd: 2*af2_pmhc_tcr_pae + 1*rfab_pmhc_tcr_pae + 0.5*af2_rfab_cdr3_rmsd

whichever criterion is used should already be a column in the starting poolfile.


Each process reads the pool, selects some members of the pool for refinement,
and then at the end potentially adds the refined members to the pool

parameters:

--max_pool_size: eg 500

--max_per_lineage: eg 10 (will start with 1 per lineage). Lineage is defined by a
'lineage' column in the input --poolfile

--num_parents: eg 10, the number of starting designs each process will select
from the pool.
These designs don't interact with one another, the reason to do more than 1 is just
so that the alphafold calculations can be done together to amortize the computational
cost of model compilation.

--num_mutations: eg 2, number of mutations to make

Here's the process:

1. start by picking --num_parents members from the pool

2. make --num_mutations random sequence mutations to each one, with mutations
occurring at the positions given in the 'designable_positions' column

3. alphafold-dock them (re-using alignfile and template info if present)

4. MPNN them

5. alphafold-dock them

6. save the results

7. read the new poolfile (which might have been modified by other processes in the
meantime): do any of these new designs we made make the cut? if so, add them to
the pool subject to --max_per_lineage and --max_pool_size



Example command line:

python dock_refine.py  --max_pool_size 80 \\
    --max_per_lineage 10  --num_parents 10 \\
    --num_mutations 2  --sort_tag combo_score_wtd \\
    --poolfile /home/pbradley/csdat/tcrpepmhc/amir/run408_A0201_TLMSAMTNL_pool.tsv \\
    --outfile_prefix /home/pbradley/csdat/tcrpepmhc/amir/slurm/run408/run408_A0201_TLMSAMTNL_00228

The --poolfile file should have (at least) these columns:

    * chainseq
    * lineage
    * designable_positions (comma-separated list, made by dock_design.py)
    * pmhc_tcr_pae -OR- combo_score -OR- combo_score_wtd , depends on --sort_tag

AND EITHER
    * alignfile (so we don't have to run alphafold setup again)
OR
    * organism, mhc, mhc_class, peptide, va, ja, cdr3a, vb, jb, cdr3b (for af2 setup)


email pbradley@fredhutch.org with questions

''',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)


parser.add_argument('--poolfile', required=True)
parser.add_argument('--outfile_prefix', required=True)

parser.add_argument('--num_mutations', type=int, default=2)
parser.add_argument('--num_parents', type=int, default=10)
parser.add_argument('--max_pool_size', type=int, default=200)
parser.add_argument('--max_per_lineage', type=int, default=10)

parser.add_argument('--force_tcr_pdbids_column')

parser.add_argument('--sort_tag', required=True,
                    choices = ['pmhc_tcr_pae', 'combo_score', 'combo_score_wtd'])
parser.add_argument('--debug', action='store_true')
parser.add_argument('--drop_duplicates', action='store_true')
parser.add_argument('--run_rfab', action='store_true',
                    help='Whether to run rf_antibody on the designs; if --sort_tag is '
                    'combo_score or combo_score_wtd, this is automatically done')
parser.add_argument('--num_recycle', type=int, default=3)

parser.add_argument('--model_name', default='model_2_ptm_ft_binder',
                    help='this doesnt really matter but it has to start with '
                    '"model_2_ptm_ft"')

parser.add_argument('--model_params_file',
                    help='The default is a binder-fine-tuned model that was trained '
                    'on structures and a new distillation set')

#parser.add_argument('--sort_descending', action='store_true')


args = parser.parse_args()

if args.sort_tag in ['combo_score', 'combo_score_wtd']:
    args.run_rfab = True


## more imports ####################
import sys
import pandas as pd
import numpy as np
import random
from os import system, popen, mkdir
from sys import exit
import itertools as it
from timeit import default_timer as timer
from scipy.stats import describe
from pathlib import Path
from os.path import exists, isdir
import os
import design_paths # local
design_paths.setup_import_paths()
from tcrdock.tcrdist.amino_acids import amino_acids
from filelock.filelock import FileLock

# local imports
import design_stats
import wrapper_tools

######################################################################################88
## functions
######################################################################################88

def diversify_parents(parents, num_mutations):
    ''' Returns targets
    '''
    required_cols = 'targetid chainseq designable_positions'.split()
    for col in required_cols:
        assert col in parents.columns

    dfl = []
    for _, l in parents.iterrows():
        cbs = [0] + list(it.accumulate(len(x) for x in l.chainseq.split('/')))
        oldseq = l.chainseq.replace('/','')
        flex_posl = [int(x) for x in l.designable_positions.split(',')]
        outl = l.copy()
        outl['targetid'] = l.targetid+'_v'
        mut_posl = random.choices(flex_posl, k=num_mutations)
        newseq = list(oldseq)
        for pos in mut_posl:
            new_aa = random.choice(amino_acids)
            newseq[pos] = new_aa
        newseq = ''.join(newseq)
        outl['chainseq'] = '/'.join(newseq[x:y] for x,y in zip(cbs, cbs[1:]))
        assert len(outl.chainseq) == len(l.chainseq)

        if 'cdr3a' in parents.columns:
            for col in 'cdr3a cdr3b'.split():
                old = l[col]
                start = l.chainseq.index(old)
                new = outl.chainseq[start:start+len(old)]
                outl[col] = new

        if dfl and outl['chainseq'] in set(x.chainseq for x in dfl):
            print('duplicate chainseq!', outl.targetid)
            continue
        dfl.append(outl)
    targets = pd.DataFrame(dfl)

    return targets


######################################################################################88
## main
######################################################################################88

if args.model_params_file is None:
    args.model_params_file = design_paths.AF2_BINDER_FT_PARAMS



with FileLock(args.poolfile, timeout=120, delay=0.3) as lock:
    pool = pd.read_table(args.poolfile)
    parents = pool.sample(n=args.num_parents)

required_columns = 'lineage designable_positions chainseq'.split()
required_columns.append(args.sort_tag) # need

if 'alignfile' in parents.columns and all(exists(x) for x in set(parents.alignfile)):
    RUN_AF2_SETUP = False
else:
    RUN_AF2_SETUP = True
    required_columns += 'organism mhc mhc_class peptide va ja cdr3a vb jb cdr3b'.split()
    parents['old_chainseq'] = parents.chainseq # since setup will create new


if args.force_tcr_pdbids_column is None:
    for col in ['tcr_template_pdbid', 'tcr_pdbid']:
        if col in parents.columns:
            print('reading tcr template pdbid from column:', col)
            args.force_tcr_pdbids_column = col
            break


for col in required_columns:
    assert col in parents.columns, f'Need column {col} in {args.poolfile}'

if 'targetid' not in parents.columns: # set default
    parents['targetid'] = [f'T{x:04d}' for x in range(parents.shape[0])]

else:
    parents['parent_targetid'] = parents.targetid

    if parents.targetid.value_counts().max() >1:
        parents['targetid'] = [f'{x}_{i}' for i,x in enumerate(parents.targetid)]

workdir = f'{args.outfile_prefix}_workdir/'
os.makedirs(workdir, exist_ok=True)

if RUN_AF2_SETUP:
    parents['old_chainseq'] = parents.chainseq
    parents.drop(columns=['chainseq','alignfile'], inplace=True)
    parents = td2.sequtil.setup_for_alphafold(
        parents, workdir, num_runs=1, use_opt_dgeoms=True, clobber=True,
        force_tcr_pdbids_column=args.force_tcr_pdbids_column,
        use_new_templates=True,
    )

    parents.rename(columns={'target_chainseq':'chainseq',
                             'templates_alignfile':'alignfile'}, inplace=True)
    assert all(parents.chainseq==parents.old_chainseq),\
        ('dock_refine.py is only working for cdr3 designs right now... '
         'unless we can re-use the old alignfile info. Talk to Phil if '
         'this is a limitation, cuz its fixable')



# mutate the parents to generate the targets
# note FWIW we setup for alphafold before diversifying
targets = diversify_parents(parents, args.num_mutations)
targets.to_csv(f'{workdir}start_targets.tsv', sep='\t', index=False)

# run alphafold
outprefix = f'{workdir}afold1'
start = timer()
targets = wrapper_tools.run_alphafold(
    targets, outprefix,
    num_recycle = args.num_recycle,
    model_name = args.model_name,
    model_params_file = args.model_params_file,
    dry_run = args.debug,
    ignore_identities = True,
)
af2_time = timer()-start


targets = design_stats.compute_simple_stats(targets, extend_flex='barf')
if 'dgeom_rmsd' in targets.columns:
    targets['dgeom_rmsd_to_parent'] = targets.dgeom_rmsd

targets.to_csv(f'{workdir}afold1_results.tsv', sep='\t', index=False)

# run mpnn
outprefix = f'{workdir}_mpnn'
start = timer()
targets = wrapper_tools.run_mpnn(
    targets,
    outprefix,
    extend_flex='barf',
    dry_run=args.debug,
)
mpnn_time = timer()-start

# run alphafold again
outprefix = f'{workdir}_afold2'
start = timer()
targets = wrapper_tools.run_alphafold(
    targets, outprefix,
    num_recycle=args.num_recycle,
    model_name = args.model_name,
    model_params_file = args.model_params_file,
    dry_run = args.debug,
    ignore_identities = True,
)
af2_time += timer()-start

# compute stats again. this should compute docking rmsd to the original mpnn dock
targets = design_stats.compute_simple_stats(targets, extend_flex='barf')
targets['af2_time'] = af2_time/targets.shape[0]
targets['mpnn_time'] = mpnn_time/targets.shape[0]

if args.run_rfab:
    outprefix = f'{workdir}_rf2'
    start = timer()
    # will have cols 'rfab_pbind' 'rfab_pmhc_tcr_pae'
    rf_targets = wrapper_tools.run_rf_antibody_on_designs(targets, outprefix)
    rf2_time = timer()-start
    targets['rf2_time'] = rf2_time/targets.shape[0]

    for col in 'model_pdbfile rfab_pbind rfab_pmhc_tcr_pae'.split():
        targets['rf2_'+col] = rf_targets[col]

    df = design_stats.compare_models(targets, rf_targets)
    for col in 'dgeom_rmsd cdr3_rmsd cdr_rmsd'.split():
        targets['rf2_'+col] = df[col]
    targets['cdr_seq'] = df['model1_cdr_seq']

    targets['combo_score'] = (targets.pmhc_tcr_pae +
                              targets.rf2_rfab_pmhc_tcr_pae +
                              targets.rf2_cdr3_rmsd)

    targets['combo_score_wtd'] = (2.0 * targets.pmhc_tcr_pae +
                                  1.0 * targets.rf2_rfab_pmhc_tcr_pae +
                                  0.5 * targets.rf2_cdr3_rmsd)


targets.to_csv(f'{workdir}afold2_results.tsv', sep='\t', index=False)


# now we have to maybe add some of these to the pool
with FileLock(args.poolfile, timeout=120, delay=0.2) as lock:
    pool = pd.read_table(args.poolfile)
    print('read old pool:', pool.shape, describe(pool[args.sort_tag]))

    pool = pd.concat([pool, targets]).sort_values(args.sort_tag)
    if args.drop_duplicates:
        pool.drop_duplicates(subset=['chainseq'], inplace=True)
    pool['lcount'] = pool.groupby('lineage').cumcount()

    pool = pool[pool.lcount < args.max_per_lineage].head(args.max_pool_size)

    pool.to_csv(args.poolfile, sep='\t', index=False)

print('wrote new pool', describe(pool[args.sort_tag]))


print('DONE')




