######################################################################################88
import argparse

parser = argparse.ArgumentParser(
    description = "Read the <outprefix>_final.tsv file created by run_prediction.py "
    "and add a column named pmhc_tcr_pae that records the predicted pairwise accuracy "
    "measure PAE (predicted aligned error) averaged over all pMHC-TCR residue pairs.",
    epilog = f'''Example command lines:

python add_pmhc_tcr_pae_to_tsvfile.py  --infile test_final.tsv --outfile test_final_w_pae.tsv
''',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument('--infile', required=True,
                    help='TSV formatted file with info output from run_prediction.py')
parser.add_argument('--outfile', required=True,
                    help='Filename for the output file. Will not overwrite if it '
                    'already exists unless the --clobber option is given')
parser.add_argument('--model_name', help='Parameters codename (like "model_2_ptm") '
                    ' that was used for the model of interest. If not provided, '
                    'will try to autodetect.')
parser.add_argument('--clobber', action='store_true',
                    help='Overwrite --outfile if it already exists')

args = parser.parse_args()

import pandas as pd
import os
from os.path import exists
from pathlib import Path
import sys

if not exists(args.infile):
    print(f'ERROR The input file {args.infile} does not exist.')
    sys.exit(1)

if exists(args.outfile) and not args.clobber:
    print(f'ERROR The output file {args.outfile} already exists and --clobber is not '
          'specified.')
    sys.exit(1)


results = pd.read_table(args.infile)

model_names = [x[:-8] for x in results.columns
               if x.endswith('_pae_0_1')]

if args.model_name is None:
    model_name = model_names[0]
else:
    model_name = args.model_name

print(f'Calculating pmhc_tcr_pae for model: {model_name}')
assert model_name in model_names


inter_paes = []

for _, l in results.iterrows():
    cs = l.target_chainseq.split('/')
    num_chains = len(cs)
    assert num_chains in [4,5] # mhc class 1 or 2

    pmhc_chains = range(num_chains-2)
    tcr_chains = range(num_chains-2, num_chains)

    inter_pae = 0.
    for i in pmhc_chains:
        nres_i = len(cs[i])
        for j in tcr_chains:
            nres_j = len(cs[j])
            pae_ij = l[f'{model_name}_pae_{i}_{j}']
            pae_ji = l[f'{model_name}_pae_{j}_{i}']
            inter_pae += nres_i * nres_j * (pae_ij + pae_ji)
    nres_pmhc = sum(len(cs[x]) for x in pmhc_chains)
    nres_tcr = sum(len(cs[x]) for x in tcr_chains)
    inter_pae /= 2*nres_pmhc*nres_tcr
    inter_paes.append(inter_pae)

results['model_name'] = model_name
results['pmhc_tcr_pae'] = inter_paes
pdbcol = f'{model_name}_pdb_file'
if pdbcol in results.columns:
    results['model_pdbfile'] = results[pdbcol]

results.to_csv(args.outfile, sep='\t', index=False)
print('made:', args.outfile)
