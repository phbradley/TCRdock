''' The idea is to take a starting design or designs and iterate:

* make random mutations to form a population of variants
* alphafold to get "new" loop structures
* mpnn to get new loop sequences
* alphafold to get new loop structures and inter-paes
* select top n_founders for the next round


inputs:

--targets:  tsvfile with columns: (targetid), chainseq, alignfile,
    (peptide_residues)
--num_mutations: number of mutations per variant
--num_variants: number of variants per starting design
--num_founders: number of good designs to seed the next design round
--num_rounds: number of rounds to run


The designable positions are the ones that don't align with the template

'''

import argparse

parser = argparse.ArgumentParser(
    description="iterative loop design refinement")

parser.add_argument('--targets', required=True)
parser.add_argument('--outfile_prefix', required=True)


parser.add_argument('--num_mutations', type=int, default=2)
parser.add_argument('--num_variants', type=int, default=10)
parser.add_argument('--num_founders', type=int, default=2)
parser.add_argument('--num_rounds', type=int, default=5)
parser.add_argument('--extend_flex', type=int, default=1)
parser.add_argument('--sort_tag', default='peptide_loop_pae')
parser.add_argument('--sort_descending', action='store_true')
parser.add_argument('--random_founders', action='store_true')


args = parser.parse_args()


required_columns = ('targetid chainseq model_pdbfile template_0_template_pdbfile '
                    'template_0_target_to_template_alignstring').split()


## more imports ####################
import sys
from sys import exit
import itertools as it
from pathlib import Path
from os.path import exists, isdir
import os
import design_paths # local
design_paths.setup_import_paths()
from tcrdock.tcrdist.amino_acids import amino_acids
import pandas as pd
import numpy as np
import random
from os import system, popen, mkdir
#from glob import glob
#from collections import Counter
#import scipy
#import json
# local imports
from design_stats import compute_stats, get_designable_positions
from wrapper_tools import run_alphafold, run_mpnn


######################################################################################88
## functions
######################################################################################88

def diversify_founders(founders, num_mutations, num_variants):
    ''' Returns targets
    '''
    required_cols = (
        'targetid chainseq template_0_target_to_template_alignstring').split()
    for col in required_cols:
        assert col in founders.columns

    dfl = []
    for _, l in founders.iterrows():
        cbs = [0] + list(it.accumulate(len(x) for x in l.chainseq.split('/')))
        oldseq = l.chainseq.replace('/','')
        flex_posl = get_designable_positions(
            row=l, extend_flex=args.extend_flex)
        for v in range(num_variants):
            outl = l.copy()
            outl['targetid'] = f'{l.targetid}_v{v}'
            mut_posl = random.choices(flex_posl, k=num_mutations)
            newseq = list(oldseq)
            for pos in mut_posl:
                new_aa = random.choice(amino_acids)
                newseq[pos] = new_aa
            newseq = ''.join(newseq)
            outl['chainseq'] = '/'.join(newseq[x:y] for x,y in zip(cbs, cbs[1:]))
            assert len(outl.chainseq) == len(l.chainseq)
            if dfl and outl['chainseq'] in set(x.chainseq for x in dfl):
                print('duplicate chainseq!', outl.targetid)
                continue
            dfl.append(outl)
    targets = pd.DataFrame(dfl)

    return targets


######################################################################################88
## main
######################################################################################88

founders = pd.read_table(args.targets)

for col in required_columns:
    assert col in founders.columns, f'Need column {col} in {args.targets}'

if 'targetid' not in founders.columns: # set default
    founders['targetid'] = [f'T{x:04d}' for x in range(founders.shape[0])]

assert founders.targetid.value_counts().max() == 1 # no dups

if args.random_founders:
    assert args.num_founders<founders.shape[0]
    founders = founders.sample(args.num_founders)

for r in range(args.num_rounds):

    workdir = f'{args.outfile_prefix}_round_{r:03d}/'
    if not exists(workdir):
        mkdir(workdir)

    # mutate the founders to generate the targets
    targets = diversify_founders(founders, args.num_mutations, args.num_variants)
    targets.to_csv(f'{workdir}start_targets.tsv', sep='\t', index=False)

    # run alphafold
    outprefix = f'{workdir}afold1'
    targets = run_alphafold(targets, outprefix)
    targets = compute_stats(targets) # like rmsds, recovery?
    targets.to_csv(f'{workdir}afold1_results.tsv', sep='\t', index=False)

    # run mpnn
    outprefix = f'{workdir}mpnn'
    targets = run_mpnn(targets, outprefix, extend_flex=args.extend_flex)
    targets.to_csv(f'{workdir}mpnn_results.tsv', sep='\t', index=False)

    # run alphafold again
    outprefix = f'{workdir}afold2'
    targets = run_alphafold(targets, outprefix)
    targets = compute_stats(targets) # like rmsds, recovery?
    targets.to_csv(f'{workdir}afold2_results.tsv', sep='\t', index=False)

    founders = targets.sort_values(
        args.sort_tag, ascending=not args.sort_descending).head(args.num_founders)

    founders.to_csv(f'{workdir}next_founders.tsv', sep='\t', index=False)

print('DONE')




