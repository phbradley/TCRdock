import sys
import os
from os.path import exists
from glob import glob
import pickle
from collections import OrderedDict
import design_paths # local import
design_paths.setup_import_paths()

######################################################################################88
## template script dependencies
from sys import exit
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List

import numpy as np
#import pandas as pd

from alphafold.data import templates
import haiku as hk # for splitting params

import random

## predict complexes script dependencies (duplicates removed)

from timeit import default_timer as timer


from alphafold.common import residue_constants
from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import feature_processing
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.data import parsers # needed for NEW_ALPHAFOLD
#from alphafold.data.tools import hhsearch
#from alphafold.relax import relax ## this gives an error due to missing pdbfixer...

# from https://sbgrid.org/wiki/examples/alphafold2
# TF_FORCE_UNIFIED_MEMORY=1
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
# XLA_PYTHON_CLIENT_ALLOCATOR=platform

# from amir:
#os.environ["TF_FORCE_UNIFIED_MEMORY"] = '1'
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '2.0'

# fiddling around:

#2021-10-14 12:13:06.884915: W external/org_tensorflow/tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcusolver.so.11'; dlerror: libcusolver.so.11: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/pbradley/src/lib:/home/pbradley/src/lib:
#  518  export LD_LIBRARY_PATH=/home/pbradley/anaconda2/envs/af2/lib:$LD_LIBRARY_PATH




# CHANGE THIS (directory where "params/ folder is")
data_dir = design_paths.AF2_DATA_DIR


def load_pdb_coords(
        pdbfile,
        allow_chainbreaks=False,
        allow_skipped_lines=False,
        verbose=False,
):
    ''' returns: chains, all_resids, all_coords, all_name1s
    '''

    chains = []
    all_resids = {}
    all_coords = {}
    all_name1s = {}

    if verbose:
        print('reading:', pdbfile)
    skipped_lines = False
    with open(pdbfile,'r') as data:
        for line in data:
            if (line[:6] in ['ATOM  ','HETATM'] and line[17:20] != 'HOH' and
                line[16] in ' A1'):
                if ( line[17:20] in residue_constants.restype_3to1
                     or line[17:20] == 'MSE'): # 2022-03-31 change to include MSE
                    name1 = ('M' if line[17:20] == 'MSE' else
                             residue_constants.restype_3to1[line[17:20]])
                    resid = line[22:27]
                    chain = line[21]
                    if chain not in all_resids:
                        all_resids[chain] = []
                        all_coords[chain] = {}
                        all_name1s[chain] = {}
                        chains.append(chain)
                    if line.startswith('HETATM'):
                        print('WARNING: HETATM', pdbfile, line[:-1])
                    atom = line[12:16].split()[0]
                    if resid not in all_resids[chain]:
                        all_resids[chain].append(resid)
                        all_coords[chain][resid] = {}
                        all_name1s[chain][resid] = name1

                    all_coords[chain][resid][atom] = np.array(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])])
                else:
                    print('skip ATOM line:', line[:-1], pdbfile)
                    skipped_lines = True

    # check for chainbreaks
    maxdis = 1.75
    for chain in chains:
        for res1, res2 in zip(all_resids[chain][:-1], all_resids[chain][1:]):
            coords1 = all_coords[chain][res1]
            coords2 = all_coords[chain][res2]
            if 'C' in coords1 and 'N' in coords2:
                dis = np.sqrt(np.sum(np.square(coords1['C']-coords2['N'])))
                if dis>maxdis:
                    print('ERROR chainbreak:', chain, res1, res2, dis, pdbfile)
                    if not allow_chainbreaks:
                        print('STOP: chainbreaks', pdbfile)
                        #print('DONE')
                        exit()

    if skipped_lines and not allow_skipped_lines:
        print('STOP: skipped lines:', pdbfile)
        #print('DONE')
        exit()

    return chains, all_resids, all_coords, all_name1s


def fill_afold_coords(
        chain_order,
        all_resids,
        all_coords,
):
    ''' returns: all_positions, all_positions_mask

    these are 'atom37' coords (not 'atom14' coords)

    '''
    assert residue_constants.atom_type_num == 37 #HACK/SANITY
    crs = [(chain,resid) for chain in chain_order for resid in all_resids[chain]]
    num_res = len(crs)
    all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
    all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num],
                                  dtype=np.int64)
    for res_index, (chain,resid) in enumerate(crs):
        pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
        mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)
        for atom_name, xyz in all_coords[chain][resid].items():
            x,y,z = xyz
            if atom_name in residue_constants.atom_order.keys():
                pos[residue_constants.atom_order[atom_name]] = [x, y, z]
                mask[residue_constants.atom_order[atom_name]] = 1.0
            elif atom_name != 'NV': # PRO NV OK to skip
                # this is just debugging/verbose output:
                name = atom_name[:]
                while name[0] in '123':
                    name = name[1:]
                if name[0] != 'H':
                    print('unrecognized atom:', atom_name, chain, resid)
            # elif atom_name.upper() == 'SE' and res.get_resname() == 'MSE':
            #     # Put the coordinates of the selenium atom in the sulphur column.
            #     pos[residue_constants.atom_order['SD']] = [x, y, z]
            #     mask[residue_constants.atom_order['SD']] = 1.0

        all_positions[res_index] = pos
        all_positions_mask[res_index] = mask
    return all_positions, all_positions_mask

def fill_in_cbeta_coords(
        pos,
        all_positions,
        all_positions_mask,
):
    from numpy.linalg import norm
    from numpy import dot, cross
    mean_cb_coords = np.array([-0.53529231, -0.76736402,  1.20778869])

    N  = residue_constants.atom_order['N']
    CA = residue_constants.atom_order['CA']
    C  = residue_constants.atom_order['C']
    CB = residue_constants.atom_order['CB']

    masks = all_positions_mask[pos, [N,CA,C]]
    if masks.sum() < 2.5:
        print('fill_in_cbeta_coords: missing N,CA,C', masks)
        return

    n  = all_positions[pos,N ]
    ca = all_positions[pos,CA]
    c  = all_positions[pos,C ]

    origin = ca
    x = (n-ca)/norm(n-ca)
    y = (c-ca)
    y -= x * dot(x,y)
    y /= norm(y)
    z = cross(x,y)
    assert abs(norm(x)-1)<1e-3
    assert abs(norm(y)-1)<1e-3
    assert abs(norm(z)-1)<1e-3
    assert abs(dot(x,y)<1e-3)
    assert abs(dot(z,y)<1e-3)
    assert abs(dot(x,z)<1e-3)

    new_cb = origin + (mean_cb_coords[0] * x +
                       mean_cb_coords[1] * y +
                       mean_cb_coords[2] * z)

    if all_positions_mask[pos, CB] > 0.5: #CB already exists
        cb = all_positions[pos,CB]
        xc = dot(cb-origin, x)
        yc = dot(cb-origin, y)
        zc = dot(cb-origin, z)
        dev = np.sqrt(((cb-new_cb)**2).sum())
        print(f'cb_xyz {xc:7.3f} {yc:7.3f} {zc:7.3f} dev: {dev:7.4f}')
    else:
        print('fill_cb:', pos)
        all_positions[pos, CB] = new_cb
        all_positions_mask[pos, CB] = 1.0

    return

    #return np.array([xc, yc, zc]) # for collecting stats


def run_alphafold_prediction(
        query_sequence: str,
        msa: list,
        deletion_matrix: list,
        chainbreak_sequence: str,
        template_features: dict,
        model_runners: dict,
        out_prefix: str,
        #crop_size=None,
        dump_pdbs=True,
        dump_metrics=True,
        force_residue_index=None,
):
    '''msa should be a list. If single seq is provided, it should be a list of str.

    returns a dictionary with keys= model_name, values= dictionary
    indexed by metric_tag
    '''

    fake_descriptions = [f'msa_seq{x}' for x in range(len(msa))]
    msa_class = parsers.Msa(sequences=msa, deletion_matrix=deletion_matrix,
                            descriptions=fake_descriptions)
    # gather features for running with only template information
    feature_dict = {
        **pipeline.make_sequence_features(sequence=query_sequence,
                                          description="none",
                                          num_res=len(query_sequence)),
        **pipeline.make_msa_features(msas=[msa_class]),
        # **pipeline.make_msa_features(msas=[msa],
        #                              deletion_matrices=[deletion_matrix]),
        **template_features
    }

    if force_residue_index is None:
        # add big enough number to residue index to indicate chain breaks

        # Ls: number of residues in each chain
        # Ls = [ len(split) for split in chainbreak_sequence.split('/') ]
        Ls = [ len(split) for split in chainbreak_sequence.split('/') ]
        idx_res = feature_dict['residue_index']
        L_prev = 0
        for L_i in Ls[:-1]:
            idx_res[L_prev+L_i:] += 200
            L_prev += L_i
        feature_dict['residue_index'] = idx_res
    else:
        old_idx = feature_dict['residue_index']
        print('old:', old_idx.shape, old_idx.dtype, type(old_idx), 'new:',
              force_residue_index.shape, force_residue_index.dtype,
              type(force_residue_index))
        feature_dict['residue_index'] = force_residue_index.astype(old_idx.dtype)

    model_names = list(model_runners.keys())
    if any('multimer' in x for x in model_names):
        assert all('multimer' in x for x in model_names)
        # could be a problem if some are multimer models and some aren't
        chain_id = 'A'
        chain_features = pipeline_multimer.convert_monomer_features(
            feature_dict, chain_id)
        all_chain_features = OrderedDict()
        all_chain_features[chain_id] = chain_features

        all_chain_features = pipeline_multimer.add_assembly_features(
            all_chain_features)

        feature_dict = feature_processing.pair_and_merge(
            all_chain_features=all_chain_features,
            is_prokaryote=False,
        )

        # Pad MSA to avoid zero-sized extra_msa.
        feature_dict = pipeline_multimer.pad_msa(feature_dict, 512)


    all_metrics = predict_structure(
        out_prefix, feature_dict, model_runners,# crop_size=crop_size,
        dump_pdbs=dump_pdbs, dump_metrics=dump_metrics,
    )

    #np.save('{}_plddt.npy'.format(out_prefix), plddts['model_1'])

    return all_metrics


def predict_structure(
        prefix,
        feature_dict,
        model_runners,
        random_seed=0,
        #crop_size=None,
        dump_pdbs=True,
        dump_metrics=True,
):
    """Predicts structure using AlphaFold for the given sequence.

    returns a dictionary with keys= model_name, values= dictionary
    indexed by metric_tag
    """

    #model_params, model_runner_1, model_runner_3 = model_info

    # Run the models.
    #plddts = []
    unrelaxed_pdb_lines = []
    relaxed_pdb_lines = []
    model_names = []

    metric_tags = 'plddt ptm predicted_aligned_error'.split()

    all_metrics = {} # eventual return value

    metrics = {} # stupid duplication

    for model_name, model_runner in model_runners.items():
        start = timer()
        print(f"running {model_name}")
        # swap params to avoid recompiling
        # note: models 1,2 have diff number of params compared to models 3,4,5
        # if any(str(m) in model_name for m in [1,2]): model_runner = model_runner_1
        # if any(str(m) in model_name for m in [3,4,5]): model_runner = model_runner_3
        # model_runner.params = params

        processed_feature_dict = model_runner.process_features(
            feature_dict, random_seed=random_seed)#, crop_size=crop_size)

        # random_seed is now needed "as MSA sampling happens inside the Multimer model."
        prediction_result = model_runner.predict(processed_feature_dict, random_seed=0)

        unrelaxed_protein = protein.from_prediction(
            features=processed_feature_dict,
            result=prediction_result,
            #b_factors=plddt_b_factors,
            remove_leading_feature_dimension=not model_runner.multimer_mode)
        # unrelaxed_protein = protein.from_prediction(
        #     processed_feature_dict, prediction_result)
        unrelaxed_pdb_lines.append(protein.to_pdb(unrelaxed_protein))
        model_names.append(model_name)

        all_metrics[model_name] = {}
        for tag in metric_tags:
            result = prediction_result.get(tag, None)
            metrics.setdefault(tag, []).append(result)
            if result is not None:
                all_metrics[model_name][tag] = result

        print(f"{model_name} pLDDT: {np.mean(prediction_result['plddt'])} "
              f"Time: {timer() - start}")

    # rerank models based on predicted lddt
    plddts = metrics['plddt']
    lddt_rank = np.mean(plddts,-1).argsort()[::-1]
    #plddts_ranked = {}
    for n, r in enumerate(lddt_rank):
        model_name = model_names[r]
        print(f"model_{n+1} {model_name} {np.mean(plddts[r])}")

        if dump_pdbs:
            #unrelaxed_pdb_path = f'{prefix}_model_{n+1}_{model_name}.pdb'
            unrelaxed_pdb_path = f'{prefix}_model_1_{model_name}.pdb' # predictable!
            with open(unrelaxed_pdb_path, 'w') as f: f.write(unrelaxed_pdb_lines[r])
            all_metrics[model_name]['pdbfile'] = unrelaxed_pdb_path

        #plddts_ranked[f"model_{n+1}"] = plddts[r]

        if dump_metrics:
            #metrics_prefix = f'{prefix}_model_{n+1}_{model_name}'
            metrics_prefix = f'{prefix}_model_1_{model_name}' # more predictable now
            for tag in metric_tags:
                m = metrics[tag][r]
                if m is not None:
                    fname = f'{metrics_prefix}_{tag}.npy'
                    np.save(fname, m)
                    all_metrics[model_name][f'{tag}file'] = fname

    return all_metrics
    #return plddts_ranked



# def load_model_params(
#         num_models,
#         num_recycle = 3,
# ):
#     ''' obsolete

#     returns model_info = (model_params, model_runner_1, model_runner_3)
#     runners might be None if not used
#     '''

#     # loading model params in memory
#     model_params = {}
#     model_runner_1, model_runner_3 = None, None

#     for model_name in ["model_1","model_2","model_3","model_4","model_5"][:num_models]:
#         if model_name not in model_params:
#             print('config:', model_name)
#             model_config = config.model_config(model_name)
#             model_config.data.eval.num_ensemble = 1

#             model_config.data.common.num_recycle = num_recycle
#             model_config.model.num_recycle = num_recycle

#             model_params[model_name] = data.get_model_haiku_params(
#                 model_name=model_name, data_dir=data_dir)

#             if model_name == "model_1":
#                 model_runner_1 = model.RunModel(model_config, model_params[model_name])
#             if model_name == "model_3":
#                 model_runner_3 = model.RunModel(model_config, model_params[model_name])
#     return (model_params, model_runner_1, model_runner_3)

def load_model_runners(
        model_names,
        crop_size,
        num_recycle = 3,
        num_ensemble = 1,
        model_params_file = None,
        model_params_files = None,
        resample_msa_in_recycling = True,
        small_msas = True,
):
    if model_params_file is not None:
        assert len(model_names) == 1
        model_params_files = [model_params_file]
    elif model_params_files is None:
        model_params_files = [None]*len(model_names)

    assert len(model_names) == len(model_params_files)

    model_runners = OrderedDict()
    for model_name, model_params_file in zip(model_names, model_params_files):
        print('config:', model_name)
        af_model_name = (model_name[:model_name.index('_ft')] if '_ft' in model_name
                         else model_name)
        model_config = config.model_config(af_model_name)

        if 'multimer' in model_name:
            model_config.model.num_ensemble_eval = num_ensemble
        else:
            model_config.data.eval.crop_size = crop_size
            model_config.data.eval.num_ensemble = num_ensemble
            model_config.data.common.num_recycle = num_recycle
            model_config.model.num_recycle = num_recycle
            if small_msas:
                print('load_model_runners:: small_msas==True setting small',
                      'max_extra_msa and max_msa_clusters')
                model_config.data.common.max_extra_msa = 1 #############
                model_config.data.eval.max_msa_clusters = 5 ###############
            if not resample_msa_in_recycling:
                model_config.data.common.resample_msa_in_recycling = False
                model_config.model.resample_msa_in_recycling = False


        if model_params_file != 'classic' and model_params_file is not None:
            print('loading', model_name, 'params from file:', model_params_file)
            with open(model_params_file, 'rb') as f:
                model_params = pickle.load(f)

            model_params, other_params = hk.data_structures.partition(
                lambda m, n, p: m[:9] == "alphafold", model_params)
            print('ignoring other_params:', other_params)

        else:
            assert '_ft' not in model_name
            model_params = data.get_model_haiku_params(
                model_name=model_name, data_dir=data_dir)

        model_runners[model_name] = model.RunModel(
            model_config, model_params)
    return model_runners

