'''
'''

required_cols = ('targetid chainseq template_0_template_pdbfile model_pdbfile '
                 'template_0_target_to_template_alignstring'.split())

import argparse
parser = argparse.ArgumentParser(description="setup for peptide x-scanning")

parser.add_argument('--targets', required=True)
parser.add_argument('--outfile', required=True)
parser.add_argument('--extend_flex', type=int, default=1)

args = parser.parse_args()

import pandas as pd
targets = pd.read_table(args.targets)

for col in required_cols:
    assert col in targets.columns

assert targets.targetid.value_counts().max() == 1 # no duplicates


######################################################################################88
# more imports
import design_stats
import design_paths
design_paths.setup_import_paths()
import tcrdock as td2
from tcrdock.tcrdist.amino_acids import amino_acids
from os.path import exists
import itertools as it
import numpy as np

######################################################################################88


def setup_for_xscans(
        targets,
        outfile,
        mask_char='-',
):
    if 'peptide' not in targets.columns:
        targets['peptide'] = targets.chainseq.str.split('/').str.get(-3)

    dfl = []
    for _,l in targets.iterrows():
        peplen = len(l.peptide)
        cbs = [0] + list(it.accumulate(len(x) for x in l.chainseq.split('/')))
        _, nres_mhc, nres_pmhc, _, nres = cbs
        sequence = l.chainseq.replace('/','')
        full_alignstring = ';'.join(f'{i}:{i}' for i in range(nres))

        loop_inds = design_stats.get_designable_positions(
            row=l, extend_flex=args.extend_flex)

        alt_template_seq = list(sequence)
        for pos in it.chain(range(nres_mhc, nres_pmhc), loop_inds):
            alt_template_seq[pos] = mask_char
        alt_template_seq = ''.join(alt_template_seq)

        for pos in range(peplen): # the x-scan pos
            for aa in amino_acids:
                if aa == l.peptide[pos] and pos>0:
                    continue # only do the wt peptide once
                new_peptide = ''.join(
                    aa if i==pos else x for i,x in enumerate(l.peptide))

                dfl.append(dict(
                    targetid = f'{l.targetid}_xscan_{pos}_{aa}',
                    design_targetid = l.targetid,
                    chainseq = l.chainseq.replace(l.peptide, new_peptide),
                    template_0_template_pdbfile = l.model_pdbfile,
                    template_0_target_to_template_alignstring = full_alignstring,
                    template_0_alt_template_sequence = alt_template_seq,
                    peptide = new_peptide,
                    wt_peptide = l.peptide,
                    pos = pos,
                    aa = aa,
                ))
    pd.DataFrame(dfl).to_csv(outfile, sep='\t', index=False)
    print('made:', outfile)

setup_for_xscans(targets, args.outfile)
