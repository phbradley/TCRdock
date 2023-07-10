######################################################################################88
import argparse

required_columns = 'organism mhc_class mhc peptide va ja cdr3a vb jb cdr3b'.split()

parser = argparse.ArgumentParser(
    description = "Create <output_dir>/targets.tsv file and associated input files "
    "for AlphaFold modeling with the run_alphafold_predictions.py script",
    epilog = f'''The --targets_tsvfile file should contain the columns

    organism = 'mouse' or 'human'
    mhc_class = 1 or 2
    mhc = the MHC allele, e.g. 'A*02:01' or 'H2Db'
    peptide = the peptide sequence, for MHC class 2 should have length exactly 11
              (the 9 residue core plus 1 residue on either side)
    va = V-alpha gene
    ja = J-alpha gene
    cdr3a = CDR3-alpha sequence, starts with C, ends with the F/W/etc right before the
            GXG sequence in the J gene
    vb = V-beta gene
    jb = J-beta gene
    cdr3b = CDR3-beta sequence, starts with C, ends with the F/W/etc right before the
            GXG sequence in the J gene

    Check out the input files in the examples below.

Example command lines:

python setup_for_alphafold.py --targets_tsvfile examples/benchmark/single_target.tsv \\
    --output_dir test_setup_single

or

python setup_for_alphafold.py --targets_tsvfile examples/benchmark/full_benchmark.tsv \\
    --output_dir test_setup_full_benchmark --benchmark

''',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument('--targets_tsvfile', required=True,
                    help='TSV formatted file with info on modeling targets')
parser.add_argument('--output_dir', required=True,
                    help='directory where the outputs will go; preferably empty '
                    'or nonexistent (will create)')
parser.add_argument('--num_runs', type=int, default=3,
                    help='Number of alphafold runs per target (default is 3)')
parser.add_argument('--benchmark', action='store_true',
                    help='Exclude sequence-similar templates (for benchmarking)')
parser.add_argument('--maintain_relative_paths', action='store_true',
                    help='Keep the file paths in the targets and alignments files '
                    'relative (assuming --output_dir is a relative path), rather '
                    'than trying to resolve to absolute paths')
parser.add_argument('--exclude_pdbids_column',
                    help='Column in the --targets_tsvfile file with comma-separated '
                    'lists of pdbfiles to exclude from modeling')
parser.add_argument('--new_docking', action='store_true',
                    help='Use a new, unpublished approach for constructing the '
                    'TCR:pMHC docking geometry in the AlphaFold templates. '
                    'This only requires 1 run per target (ie, it is 3x faster than '
                    'the default) and preliminary tests of the fine-tuned version '
                    'are promising.')

args = parser.parse_args()

import pandas as pd
import tcrdock
import os
from pathlib import Path
import sys

if args.benchmark:
    exclude_self_peptide_docking_geometries = True
    min_single_chain_tcrdist = 36
    min_pmhc_peptide_mismatches = 3
    min_dgeom_peptide_mismatches = 3
    min_dgeom_paired_tcrdist = 48.5
    min_dgeom_singlechain_tcrdist = 0.5
else:
    exclude_self_peptide_docking_geometries = False
    min_single_chain_tcrdist = -1
    min_pmhc_peptide_mismatches = -1
    min_dgeom_peptide_mismatches = -1
    min_dgeom_paired_tcrdist = -1
    min_dgeom_singlechain_tcrdist = -1

# load the modeling targets
targets = pd.read_table(args.targets_tsvfile)
missing = [col for col in required_columns if col not in targets.columns]
if missing:
    print('ERROR --targets_tsvfile is missing required columns:', missing)
    print('see --help message for details on required columns')
    exit()

# check the gene names
ok = tcrdock.sequtil.check_genes_for_modeling(targets)

if not ok:
    print(f'ERROR some of the genes in {args.targets_tsvfile} are problematic,'
          ' see error messages above')
    sys.exit()

# make the output dir
output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

if not args.maintain_relative_paths:
    # since we are saving filenames to the targets/templates files, we want these
    # to work from wherever AlphaFold is eventually run, so try to make them
    # absolute file paths
    output_dir = str(Path(output_dir).resolve())

if not output_dir.endswith('/'):
    output_dir += '/' # this will cause issues on windows but code later assumes it...

num_runs = 1 if args.new_docking else args.num_runs

# run the setup code
tcrdock.sequtil.setup_for_alphafold(
    targets, output_dir, clobber=True,
    min_single_chain_tcrdist=min_single_chain_tcrdist,
    exclude_self_peptide_docking_geometries=exclude_self_peptide_docking_geometries,
    min_pmhc_peptide_mismatches=min_pmhc_peptide_mismatches,
    num_runs = num_runs,
    min_dgeom_peptide_mismatches=min_dgeom_peptide_mismatches, #
    min_dgeom_paired_tcrdist = min_dgeom_paired_tcrdist, #
    min_dgeom_singlechain_tcrdist = min_dgeom_singlechain_tcrdist,
    exclude_pdbids_column = args.exclude_pdbids_column,
    use_opt_dgeoms = args.new_docking,
)
