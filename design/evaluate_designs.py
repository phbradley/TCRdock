''' Evaluate some loop designs by

* rescoring with alphafold
* relaxing with rosetta

inputs:

--targets:  tsvfile with required columns:

* targetid
* chainseq
* model_pdbfile
* template_0_target_to_template_alignstring (for flex definition, w/ --extend_flex)

'''

required_cols = ('targetid chainseq model_pdbfile '
                 'template_0_target_to_template_alignstring'.split())

import os
import design_paths

import argparse
parser = argparse.ArgumentParser(description="evaluate designs")

parser.add_argument('--targets', required=True)
parser.add_argument('--outfile_prefix', required=True)
parser.add_argument('--relax_rescored_model', action='store_true')
parser.add_argument('--extend_flex', type=int, default=1)


args = parser.parse_args()

if design_paths.FRED_HUTCH_HACKS:
    assert os.environ['LD_LIBRARY_PATH'].startswith(
        '/home/pbradley/anaconda2/envs/af2/lib:'),\
        'export LD_LIBRARY_PATH=/home/pbradley/anaconda2/envs/af2/lib:$LD_LIBRARY_PATH'

## more imports ####################
design_paths.setup_import_paths()
import sys
from sys import exit
import copy
import itertools as it
from pathlib import Path
from os.path import exists, isdir
import os
import tcrdock as td2
from tcrdock.tcrdist.amino_acids import amino_acids
import pandas as pd
import numpy as np
import random
from os import system, popen, mkdir
from glob import glob
from collections import Counter, OrderedDict, namedtuple
import scipy
import json

from design_stats import get_designable_positions, add_info_to_rescoring_row
from wrapper_tools import run_alphafold


######################################################################################88
## functions
######################################################################################88


def run_alphafold_rescoring(
        targets,
        outprefix,
        model_name='model_2_ptm',
        extend_flex=args.extend_flex,
        gapchar='-',
):
    ''' will mask out with gapchar the peptide and all the unaligned residues in the
    template_0_target_to_template_alignstring's plus extend_flex rsds on either side
    of each loop segment
    '''
    dfl = []
    for l in targets.itertuples():
        sequence = l.chainseq.replace('/','')
        nres = len(sequence)
        cbs = [0]+list(it.accumulate(len(x) for x in l.chainseq.split('/')))
        nres_mhc, nres_pmhc = cbs[1:3]

        loop_flex_posl = get_designable_positions(row=l, extend_flex=extend_flex)

        full_alignstring = ';'.join(f'{i}:{i}' for i in range(nres))

        flex_posl = loop_flex_posl + list(range(nres_mhc, nres_pmhc)) # all of pep

        alt_seq = ''.join(gapchar if i in flex_posl else x
                          for i,x in enumerate(sequence))
        dfl.append(dict(
            targetid = f'{l.targetid}_rescore',
            chainseq = l.chainseq,
            template_0_template_pdbfile = l.model_pdbfile,
            template_0_target_to_template_alignstring = full_alignstring,
            template_0_alt_template_sequence = alt_seq,
        ))
    rescore_targets = pd.DataFrame(dfl)
    rescored_targets = run_alphafold(
        rescore_targets, outprefix, model_name=model_name).set_index('targetid')

    # compute stats using the paes/plddts for the rescoring run
    # add those to targets df
    #
    dfl = []
    for _, l in targets.iterrows():
        l2 = rescored_targets.loc[l.targetid+'_rescore']
        # pass 0 here since we already extended when we set up alt_template_sequence
        l2 = add_info_to_rescoring_row(l2, model_name, extend_flex=0)
        if hasattr(l,'peptide'):
            assert l.peptide == l2.peptide
        if hasattr(l,'loop_seq'):
            assert l.loop_seq == l2.loop_seq

        outl = l.copy()
        outl['rescore_peptide_plddt'] = l2.peptide_plddt
        outl['rescore_loop_plddt'] = l2.loop_plddt
        outl['rescore_peptide_loop_pae'] = l2.peptide_loop_pae
        outl['rescore_peptide'] = l2.peptide
        outl['rescore_loop_seq'] = l2.loop_seq
        outl['rescore_model_pdbfile'] = l2.model_pdbfile
        dfl.append(outl)

    return pd.DataFrame(dfl)

def run_rosetta_relax(targets, outprefix, ex_flags = False):
    PY = design_paths.PYROSETTA_PYTHON
    EXE = design_paths.path_to_design_scripts / 'relax_af2_designs.py'
    assert exists(PY) and exists(EXE)

    xargs = ' --mute '
    if ex_flags:
        xargs += ' --ex1 --ex2 '

    targets_file = outprefix+'_targets.tsv'
    relax_targets = targets.copy()
    if args.relax_rescored_model:
        print('using rescored models for relax inputs')
        print(relax_targets.rescore_model_pdbfile.head())
        print('instead of')
        print(relax_targets.model_pdbfile.head())
        relax_targets['model_pdbfile'] = relax_targets.rescore_model_pdbfile
    relax_targets.to_csv(targets_file, sep='\t', index=False)

    cmd = (f'{PY} {EXE} {xargs} --targets {targets_file} '
           f' --outfile_prefix {outprefix} > {outprefix}.log '
           f' 2> {outprefix}.err')
    print(cmd)
    system(cmd)

    resultsfile = outprefix+'_relax_af2_designs.tsv'
    assert exists(resultsfile), 'relax_af2_designs failed! '+outprefix
    original_cols = list(targets.columns)
    results = pd.read_table(resultsfile).set_index('targetid', drop=False)
    results.rename(columns=dict(
        bound_score='relax_bound_score',
        pep_score='relax_peptide_score',
        loop_score='relax_loop_score',
        pep_loop_intxn='relax_peptide_loop_intxn',
        binding_energy_frozen='relax_binding_energy_frozen',
        seq_peptide='relax_peptide',
        seq_loop='relax_loop_seq',
    ), inplace=True)

    # add length-normalized versions of the relax scores
    results['peplen'] = results.relax_peptide.str.len()
    results['looplen'] = results.relax_loop_seq.str.len()
    results['relax_peptide_score_len_norm'] = (
        results.relax_peptide_score / results.peplen)
    results['relax_loop_score_len_norm'] = (
        results.relax_loop_score / results.looplen)
    results['relax_peptide_loop_intxn_len_norm'] = (
        results.relax_peptide_loop_intxn / (results.peplen*results.looplen))

    targets = targets.join(results, on='targetid', rsuffix='_r')
    if 'peptide' in targets.columns:
        assert all(targets.peptide == targets.relax_peptide)
    if 'loop_seq' in targets.columns:
        assert all(targets.loop_seq == targets.relax_loop_seq)
    assert all(targets.rescore_peptide == targets.relax_peptide)
    assert all(targets.rescore_loop_seq == targets.relax_loop_seq)

    new_cols = ('relax_bound_score '
                'relax_peptide_score relax_peptide_score_len_norm '
                'relax_loop_score relax_loop_score_len_norm '
                'relax_peptide_loop_intxn relax_peptide_loop_intxn_len_norm '
                'relaxed_peptide_rmsd relaxed_loop_rmsd '
                'relax_binding_energy_frozen relax_time').split()

    return targets[original_cols+new_cols]

######################################################################################88
## main
######################################################################################88

targets = pd.read_table(args.targets)

for col in required_cols:
    assert col in targets.columns, f'Need {col} column in {args.targets}'

assert targets.targetid.value_counts().max() == 1 # no dups

# run alphafold rescoring
outprefix = f'{args.outfile_prefix}_afold_rescore'
targets = run_alphafold_rescoring(targets, outprefix)

# run rosetta relax
outprefix = f'{args.outfile_prefix}_relax'
targets = run_rosetta_relax(targets, outprefix)

# write results
targets.to_csv(f'{args.outfile_prefix}_final_results.tsv', sep='\t', index=False)

print('DONE')




