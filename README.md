# TCRdock

Python tools for TCR:peptide-MHC modeling and analysis

# Core functionality

* Setup and run TCR-specialized AlphaFold simulations starting from a TSV file with
TCR, peptide, and MHC information.
* Parse a TCR:peptide-MHC ternary PDB structure and define V/J/CDR3, MHC allele, TCR
and MHC coordinate frames, and TCR:pMHC docking geometry
* Calculate distances between docking geometries ('docking RMSDs') for use in clustering/docking analysis and model evaluation

# Examples

## Parse a ternary structure PDB file

The PDB file should just contain a single copy of the ternary structure, with the
following chains (in order)

MHC-I:    (1) MHC, (2) beta-2 microglobulin, (3) peptide, (4) TCRA, (5) TCRB

or MHC-I: (1) MHC, (2) peptide, (3) TCRA, (4) TCRB

MHC-II:   (1) MHCA, (2) MHCB, (3) peptide, (4) TCRA, (5) TCRB

It's not a bad idea to "clean" the pdb file by removing extraneous ligands/waters/etc.

```
python parse_tcr_pmhc_pdbfile.py --pdbfiles examples/parsing/1qsf.pdb \
    --organism human --mhc_class 1 --out_tsvfile parse_output.tsv
```

## Setup for AlphaFold modeling a set of TCR:pMHC complexes

```
python setup_for_alphafold.py --targets_tsvfile examples/benchmark/single_target.tsv \
    --output_dir test_setup_single
```

or

```
python setup_for_alphafold.py --targets_tsvfile examples/benchmark/full_benchmark.tsv \
    --output_dir test_setup_full_benchmark --benchmark
```

Here the `--benchmark` flag tells the script to exclude nearby templates.


## Run AlphaFold modeling using outputs from the above setup

Here `$ALPHAFOLD_DATA_DIR` should point to a folder that contains the AlphaFold
`params/` folder.

This will use the very-slightly-modified version of the `alphafold` library included
with this repository (see `changes_to_alphafold.txt`). It should also run OK
with any post-Nov 2021 version of
`alphafold`, but it may not be as efficient (length changes in the targets list
will trigger recompilation of the model). You will need to run in a Python environment
that has the packages required by alphafold (or in the alphafold docker instance).
If you have an older (pre-multimer) version of alphafold, try changing the variable
`NEW_ALPHAFOLD` to `False` at the top of `predict_utils.py`.

```
python run_prediction.py --targets test_setup_single/targets.tsv \
    --outfile_prefix test_run_single --model_names model_2_ptm \
    --data_dir $ALPHAFOLD_DATA_DIR
```

or

```
python run_prediction.py --targets test_setup_full_benchmark/targets.tsv \
    --outfile_prefix test_run_full --model_names model_2_ptm \
    --data_dir $ALPHAFOLD_DATA_DIR
```
