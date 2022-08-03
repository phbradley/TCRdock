# TCRdock

Python tools for TCR:peptide-MHC modeling and analysis

# Core functionality

* Parse a TCR:peptide-MHC ternary PDB structure and define V/J/CDR3, MHC allele, TCR
and MHC coordinate frames, and TCR:pMHC docking geometry
* Given a TSV file with TCR, peptide, and MHC information, setup for running
template-based AlphaFold simulations.
* Run simple template-based AlphaFold simulations
* Calculate distances between docking geometries ('docking RMSDs') for use in clustering/docking analysis

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

