# change these as needed
FRED_HUTCH_HACKS = False
if FRED_HUTCH_HACKS:
    AF2_PYTHON = '/home/pbradley/anaconda2/envs/af2/bin/python'
    PYROSETTA_PYTHON = '/home/pbradley/anaconda2/envs/pyrosetta36/bin/python'
    MPNN_PYTHON = '/home/pbradley/anaconda2/envs/RF2atest/bin/python'
    LOOP_DESIGN_PYTHON = PYROSETTA_PYTHON # most will work here
else: # UW paths??
    AF2_PYTHON = '/home/justas/.conda/envs/mlfold-test/bin/python'
    PYROSETTA_PYTHON = '/software/conda/envs/pyrosetta/bin/python'
    MPNN_PYTHON = '/home/justas/.conda/envs/mlfold-test/bin/python'
    LOOP_DESIGN_PYTHON = AF2_PYTHON # most will work here

MPNN_SCRIPT = '/home/pbradley/gitrepos/ProteinMPNN/protein_mpnn_run.py'
##

import sys
from pathlib import Path

path_to_tcrdock = Path(__file__).parents[1].resolve() # also path to alphafold
path_to_design_scripts = Path(__file__).parents[0].resolve()

def setup_import_paths():
    if str(path_to_tcrdock) != sys.path[0]:
        sys.path.insert(0, str(path_to_tcrdock))



