''' The idea is to take a starting set of sequences and

* run alphafold to get initial models
* MPNN design the variable sequence
* run alphafold again on the MPNN-designed sequences
* compute some metrics


inputs:

--targets:  tsvfile with columns: targetid, chainseq, (alignment info)

The designable positions are the ones that don't align with the template
in the FIRST (ie "template_0_") alignment
plus --extend_flex rsds on either side of contiguous blocks

'''

required_cols = ('targetid chainseq template_0_template_pdbfile '
                 'template_0_target_to_template_alignstring'.split())

import design_paths
if design_paths.FRED_HUTCH_HACKS:
    import os
    assert os.environ['LD_LIBRARY_PATH'].startswith(
        '/home/pbradley/anaconda2/envs/af2/lib:'),\
        'export LD_LIBRARY_PATH=/home/pbradley/anaconda2/envs/af2/lib:$LD_LIBRARY_PATH'

import argparse
parser = argparse.ArgumentParser(description="alphafold loop design")

parser.add_argument('--targets', required=True)
parser.add_argument('--outfile_prefix', required=True)
parser.add_argument('--extend_flex', type=int, default=1)
parser.add_argument('--batch', type=int, help='split targets into batches of size '
                    '--batch_size and run batch number --batch>')
parser.add_argument('--batch_size', type=int, help='split targets into batches of size '
                    '--batch_size and run batch number --batch>')
parser.add_argument('--dont_sort_targets_by_length_when_batching', action='store_true')

args = parser.parse_args()

import pandas as pd

# local imports in this directory:
from design_stats import compute_stats
from wrapper_tools import run_alphafold, run_mpnn

######################################################################################88

targets = pd.read_table(args.targets)

for col in required_cols:
    assert col in targets.columns, f'Need column {col} in {args.targets}'

assert targets.targetid.value_counts().max() == 1, 'Duplicates in the targetid col!'

outfile_prefix = args.outfile_prefix
if args.batch is not None:
    assert args.batch_size is not None
    start = args.batch * args.batch_size
    stop = (args.batch+1) * args.batch_size
    print('running batch=', args.batch, 'start=', start, 'stop=', stop,
          'batch_size=', args.batch_size, 'num_targets=', targets.shape[0],
          'num_batches=', (targets.shape[0]-1)//args.batch_size + 1)
    if not args.dont_sort_targets_by_length_when_batching:
        targets['nres'] = targets.chainseq.str.len() - targets.chainseq.str.count('/')
        targets.sort_values('nres', inplace=True)
    targets = targets[start:stop].copy()
    outfile_prefix += f'_b{args.batch}'


# run alphafold
outprefix = f'{outfile_prefix}_afold1'
targets = run_alphafold(targets, outprefix)

# run mpnn
outprefix = f'{outfile_prefix}_mpnn'
targets = run_mpnn(targets, outprefix, extend_flex=args.extend_flex)

# run alphafold again
outprefix = f'{outfile_prefix}_afold2'
targets = run_alphafold(targets, outprefix)

# compute stats
targets = compute_stats(targets, extend_flex=args.extend_flex)

# write results
targets.to_csv(f'{outfile_prefix}_final_results.tsv', sep='\t', index=False)

print('DONE')
