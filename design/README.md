# Scripts for TCR loop redesign

## UNPUBLISHED/UNTESTED

These scripts operate on TSV-formatted input/output files with columns

* targetid -- unique identifier for the design
* chainseq -- '/' separated chain sequences, in the order MHC/peptide/TCRA/TCRB
* template_0_template_pdbfile -- the filename of the first modeling template 
* template_0_target_to_template_alignstring -- target-template alignment for first modeling template

and optionally

* model_pdbfile -- filename of the design model
* template_N_template_pdbfile -- (for N=1,2,3) filename of Nth modeling template
* template_N_target_to_template_alignstring -- alignment for Nth modeling template
* peptide_loop_pae -- PAE between peptide and flexible loop regions
* pdbid -- PDB ID of the design template
* loop_seq -- concatenated sequence of the flexible parts of the CDR3A and CDR3B loops
* other misc columns with design scores and rmsds

## `setup_for_cdr3_loop_design.py`

Generate a `targets` input file for input to the `loop_design.py` script.

Example:

```
python setup_for_cdr3_loop_design.py --num_runs 20000 --outfile run1_targets.tsv --template_pdbids 5m02 5tje --peptides KAVYNFATM
```


## `loop_design.py`

Run a simple alphafold loop design protocol:

* Create a model of the target loop sequence with alphafold
* Optimize the sequence of the loop using protein_mpnn
* Model the new loop sequence with alphafold

Example:

```
python loop_design.py  --targets run1_targets_batch_0.tsv --outfile_prefix run1_targets_batch_0_out
```

## `loop_refine.py`

Refine a set of top-scoring designs by iterative mutation, alphafold modeling, and protein mpnn redesign.

Example:

```
python loop_refine.py  --targets run1_top_designs.tsv --outfile_prefix run1_refine
```

## `evaluate_designs.py`

Evaluate a set of designs (for example, the top 100 by peptide_loop_pae) by
(1) alphafold-rescoring and (2) rosetta relaxing

Example:

```
python evaluate_designs.py  --targets run1_top_designs.tsv --outfile_prefix run1_eval
```

## `cluster_designs.py`

Cluster top-scoring designs based on RMSD and generate cluster pdbfiles and
heatmaps showing amino acid sequence. By default, sorts the input targets based
on the `peptide_loop_pae` column.

Example:
```
python cluster_designs.py --targets run1_results.tsv --outfile_prefix run1clust --topn 100 --split_column pdbid
```

## `setup_for_redocking.py`

Create input TSV file for the `af2_wrapper.py` script that will re-dock a set of
target designs. 

Example:
```
python cluster_designs.py --targets run1_top_designs.tsv --outfile run1redock.tsv
```
