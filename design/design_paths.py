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
    PAIRED_TCR_DB = '/home/pbradley/csdat/big_covid/big_combo_tcrs_2022-01-22.tsv'
    AF2_BINDER_FT_PARAMS = '/home/pbradley/csdat/tcrpepmhc/amir/ft_params/model_2_ptm_ft_binder_20230729.pkl'
    MPNN_SCRIPT = '/home/pbradley/gitrepos/ProteinMPNN/protein_mpnn_run.py'

else: # UW paths??
    # the old ones
    #AF2_PYTHON  = '/home/justas/.conda/envs/mlfold-test/bin/python'
    #MPNN_PYTHON = '/home/justas/.conda/envs/mlfold-test/bin/python'
    #PYROSETTA_PYTHON = '/software/conda/envs/pyrosetta/bin/python'
    # try these new ones:
    AF2_PYTHON = '/software/containers/mlfold.sif'
    MPNN_PYTHON = AF2_PYTHON
    PYROSETTA_PYTHON = '/software/containers/pyrosetta.sif'
    LOOP_DESIGN_PYTHON = AF2_PYTHON # most will work here
    RFDIFF_PYTHON = '/software/containers/RF_diffusion.sif'
    RFAB_PYTHON = '/software/containers/SE3nv.sif'

    # of should we use ??
    #RFDIFF_PYTHON = '/software/containers/RF_allatom_diffusion.sif'

    AF2_DATA_DIR = '/net/databases/lab/tcr_design/alphafold_default_params/' # where params/ lives
    # from: /home/amotmaen/for/phil/rfab/run_denovo.sh
    RFDIFF_CHK = ('/mnt/net/databases/antibody/model_weights/ab_diff/'
                  'large_crop_top5_10A_loop_Ab_TCR_4-11/models/BFF_70.pt')
    RFDIFF_SCRIPT = '/net/databases/antibody/software/rf_diffusion/run_inference.py'
    # not sure if this is analogous to the one I have at the Hutch
    # that was from jwatson3
    RFAB_SCRIPT = '/net/databases/antibody/software/rf_antibody/ab_predict.py'
    #RFAB_SCRIPT = '/home/jwatson3/projects/antibody/rf_antibody/ab_predict.py'
    # actually my version of RFAB_CHK was: /home/jwatson3/projects/antibody/rf_antibody/models/May23_noframework_nosidechains_H3swap/RFab__RFab_ab_best_May10.pt
    # but I guess this one is the same but with some training info added:
    RFAB_CHK = ('/mnt/net/databases/antibody/model_weights/rf_antibody/'
                'noframework-nosidechains-5-10-23/'
                'RFab__RFab_ab_best_May10_trainingparamsadded.pt')
    PAIRED_TCR_DB = '/net/databases/lab/tcr_design/big_combo_tcrs_2022-01-22.tsv'
    AF2_BINDER_FT_PARAMS = '/net/databases/lab/tcr_design/finetuned_alphafold_parameters/model_2_ptm_ft_binder_20230729.pkl'

    MPNN_SCRIPT = '/software/lab/mpnn/proteinmpnn/protein_mpnn_run.py'

##

import sys
from pathlib import Path

path_to_tcrdock = Path(__file__).parents[1].resolve() # also path to alphafold
path_to_design_scripts = Path(__file__).parents[0].resolve()
path_to_filelock = (Path(__file__).parents[1].resolve() /
                    'FileLock')

def setup_import_paths():
    if str(path_to_tcrdock) != sys.path[0]:
        sys.path.insert(0, str(path_to_tcrdock))
    if str(path_to_filelock) not in sys.path:
        sys.path.append(str(path_to_filelock))



