# change these as needed
FRED_HUTCH_HACKS = True
if FRED_HUTCH_HACKS:
    AF2_PYTHON = '/home/pbradley/anaconda2/envs/af2/bin/python'
    AF2_DATA_DIR = '/home/pbradley/csdat/alphafold/data/' # where params/ lives
    PYROSETTA_PYTHON = '/home/pbradley/anaconda2/envs/pyrosetta36/bin/python'
    MPNN_PYTHON = '/home/pbradley/anaconda2/envs/RF2atest/bin/python'
    LOOP_DESIGN_PYTHON = PYROSETTA_PYTHON # most will work here
    RFAB_PYTHON = '/home/pbradley/anaconda2/envs/SE3nv/bin/python'
    RFDIFF_PYTHON = '/home/pbradley/anaconda2/envs/SE3nv/bin/python'
    RFDIFF_CHK = ('/home/pbradley/gitrepos/rf_diffusion_netdbabsoft/rf_diffusion/'
                  'model_weights/BFF_70.pt')
    RFDIFF_SCRIPT = ('/home/pbradley/gitrepos/rf_diffusion_netdbabsoft/rf_diffusion/'
                     'run_inference.py')
    RFAB_SCRIPT = '/home/pbradley/gitrepos/rf_antibody/ab_predict.py'
    RFAB_CHK = ('/home/pbradley/gitrepos/rf_antibody/models/'
                'May23_noframework_nosidechains_H3swap/RFab__RFab_ab_best_May10.pt')

else: # UW paths??
    AF2_PYTHON = '/home/justas/.conda/envs/mlfold-test/bin/python'
    AF2_DATA_DIR = '/home/pbradley/csdat/alphafold/data/' # where params/ lives
    PYROSETTA_PYTHON = '/software/conda/envs/pyrosetta/bin/python'
    MPNN_PYTHON = '/home/justas/.conda/envs/mlfold-test/bin/python'
    LOOP_DESIGN_PYTHON = AF2_PYTHON # most will work here
    # from: /home/amotmaen/for/phil/rfab/run_denovo.sh
    RFDIFF_CHK = ('/mnt/net/databases/antibody/model_weights/ab_diff/'
                  'large_crop_top5_10A_loop_Ab_TCR_4-11/models/BFF_70.pt')
    RFDIFF_SCRIPT = '/net/databases/antibody/software/rf_diffusion/run_inference.py'
    RFAB_SCRIPT = '/home/jwatson3/projects/antibody/rf_antibody/ab_predict.py'
    # actually my version of RFAB_CHK was: /home/jwatson3/projects/antibody/rf_antibody/models/May23_noframework_nosidechains_H3swap/RFab__RFab_ab_best_May10.pt
    # but I guess this one is the same but with some training info added:
    RFAB_CHK = ('/mnt/net/databases/antibody/model_weights/rf_antibody/'
                'noframework-nosidechains-5-10-23/'
                'RFab__RFab_ab_best_May10_trainingparamsadded.pt')


MPNN_SCRIPT = '/home/pbradley/gitrepos/ProteinMPNN/protein_mpnn_run.py'


##

import sys
from pathlib import Path

path_to_tcrdock = Path(__file__).parents[1].resolve() # also path to alphafold
path_to_design_scripts = Path(__file__).parents[0].resolve()

def setup_import_paths():
    if str(path_to_tcrdock) != sys.path[0]:
        sys.path.insert(0, str(path_to_tcrdock))




