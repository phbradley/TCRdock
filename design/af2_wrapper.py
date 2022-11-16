''' This is a wrapper around non-MSA, template-based af2

the targets file should have columns
* targetid -- unique ID
* chainseq -- '/'-separated target chain sequences

and EITHER
* alignfile
OR
* template_N_template_pdbfile [for N=0 at least and possibly 1,2,3]
* template_N_target_to_template_alignstring [ditto]


some old help messages:
template_N_target_to_template_alignstring is ';' separated i:j, 0-indexed

transform is R;v;positions [this is optional]

each of R,v, and positions are comma separated (R is ravel order, ie by rows)
Rx+v is applied to the template coordinates at positions in positions list
positions is 0-indexed

'''
required_cols = 'targetid chainseq'.split()
######################################################################################88
import sys
from sys import exit
import os
from os.path import exists
from glob import glob
from shutil import which
import pandas as pd
import design_paths # local import
design_paths.setup_import_paths()

if design_paths.FRED_HUTCH_HACKS:
    os.environ['XLA_FLAGS']='--xla_gpu_force_compilation_parallelism=1'

    assert os.environ['LD_LIBRARY_PATH'].startswith(
        '/home/pbradley/anaconda2/envs/af2/lib:'),\
        'export LD_LIBRARY_PATH=/home/pbradley/anaconda2/envs/af2/lib:$LD_LIBRARY_PATH'

#assert which('ptxas') is not None


import argparse

parser = argparse.ArgumentParser(
    description="predict tcr:pmhc structure")

parser.add_argument('--targets', required=True)
parser.add_argument('--outfile_prefix', required=True,
                    help='Prefix that will be prepended to the output filenames')

parser.add_argument('--num_recycle', type=int, default=3)
parser.add_argument('--model_names', type=str, nargs='*', default=['model_2_ptm'])
parser.add_argument('--add_templates_to_msa', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--disallow_chainbreaks', action='store_true')
parser.add_argument('--disallow_skipped_lines', action='store_true')
parser.add_argument('--reorder_templates', action='store_true')
parser.add_argument('--ignore_identities', action='store_true')
parser.add_argument('--no_pdbs', action='store_true')
parser.add_argument('--terse', action='store_true')
parser.add_argument('--no_resample_msa', action='store_true')
parser.add_argument('--model_params_file')
parser.add_argument('--model_params_files', type=str, nargs='*')

parser.add_argument('--batch', type=int, help='split targets into batches of size '
                    '--batch_size and run batch number --batch>')
parser.add_argument('--batch_size', type=int, help='split targets into batches of size '
                    '--batch_size and run batch number --batch>')
parser.add_argument('--dont_sort_targets_by_length_when_batching', action='store_true')

args = parser.parse_args()


######################################################################################88
# more imports

import itertools as it
import numpy as np
from alphafold.data import templates
from timeit import default_timer as timer
from alphafold.common import residue_constants
from alphafold.data import templates

from predict_tcr_util import (
    load_pdb_coords, fill_afold_coords, fill_in_cbeta_coords, run_alphafold_prediction,
    load_model_runners)

if 2: # just for diagnostics
    import alphafold
    import jax
    from os import popen

    print('done importing. alphafold module location:', alphafold)

    platform = jax.local_devices()[0].platform
    hostname = popen('hostname').readlines()[0].strip()

    print('cmd:', ' '.join(sys.argv))
    print('local_device:', platform, hostname)
    sys.stdout.flush()


# from https://sbgrid.org/wiki/examples/alphafold2
# TF_FORCE_UNIFIED_MEMORY=1
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
# XLA_PYTHON_CLIENT_ALLOCATOR=platform

# from amir:
if design_paths.FRED_HUTCH_HACKS:
    os.environ["TF_FORCE_UNIFIED_MEMORY"] = '1'
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '2.0'

# fiddling around:

#2021-10-14 12:13:06.884915: W external/org_tensorflow/tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcusolver.so.11'; dlerror: libcusolver.so.11: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/pbradley/src/lib:/home/pbradley/src/lib:
#  518  export LD_LIBRARY_PATH=/home/pbradley/anaconda2/envs/af2/lib:$LD_LIBRARY_PATH




# CHANGE THIS (directory where "params/ folder is")
data_dir="/home/pbradley/csdat/alphafold/data/"

start_time = timer()



targets = pd.read_table(args.targets)
for col in required_cols:
    assert col in targets.columns, f'Need {col} in {args.targets} columns'

assert ('alignfile' in targets.columns or
        ('template_0_template_pdbfile' in targets.columns and
         'template_0_target_to_template_alignstring' in targets.columns))

assert targets.targetid.value_counts().max() == 1, 'Duplicates in the targetid col!'

outfile_prefix = args.outfile_prefix
if args.batch is not None:
    assert args.batch_size is not None
    start = args.batch * args.batch_size
    stop = (args.batch+1) * args.batch_size
    print('running batch=', args.batch, 'start=', start, 'stop=', stop,
          'batch_size=', args.batch_size, 'num_targets=', targets.shape[0],
          'num_batches=', (targets.shape[0]-1)//args.batch_size + 1)
    if not args.dont_sort_targets_by_length_when_batching:
        targets['nres'] = targets.chainseq.str.len() - targets.chainseq.str.count('/')
        targets.sort_values('nres', inplace=True)
    targets = targets[start:stop].copy()
    outfile_prefix += f'_b{args.batch}'


lens = [len(x.chainseq.replace('/',''))
        for x in targets.itertuples()]
crop_size = max(lens)
print('num_targets:', targets.shape[0], 'max_len=', crop_size, 'lens:', lens)

model_runners = load_model_runners(
    args.model_names,
    crop_size,
    model_params_file=args.model_params_file,
    model_params_files=args.model_params_files,
    resample_msa_in_recycling = not args.no_resample_msa,
    small_msas = not args.add_templates_to_msa,
    num_recycle = args.num_recycle,
)

required_align_df_cols = 'template_pdbfile target_to_template_alignstring'.split()
optional_align_df_cols = 'identities target_len template_len'.split()
other_align_df_cols = ['alt_template_sequence','transform']

final_dfl = []
for counter, targetl in targets.iterrows():
    print('START:', counter, 'of', targets.shape[0])

    # what takes precedence?
    # align info in targetl...
    if 'template_0_'+required_align_df_cols[0] in targetl.index:
        # not reading an alignfile!
        align_dfl = []
        for itmp in range(4): # up to 4 templates
            out = {}
            cols = required_align_df_cols + optional_align_df_cols + other_align_df_cols
            for col in cols:
                itmp_col = f'template_{itmp}_{col}'
                if itmp_col not in targetl.index:
                    if col in required_align_df_cols:
                        assert not out
                        break
                    else:
                        val = np.nan
                else:
                    val = targetl[itmp_col]
                out[col] = val
            if not out:
                break
            align_dfl.append(out)
        align_df = pd.DataFrame(align_dfl)
    else:
        assert 'alignfile' in targetl
        alignfile = targetl.alignfile
        align_df = pd.read_table(alignfile)

    query_chainseq = targetl.chainseq
    my_outfile_prefix = outfile_prefix+'_'+targetl.targetid

    query_sequence = query_chainseq.replace('/','')
    num_res = len(query_sequence)

    # stolen from alphafold.data.templates
    # TEMPLATE_FEATURES = {
    #     'template_aatype': np.float32,
    #     'template_all_atom_masks': np.float32,
    #     'template_all_atom_positions': np.float32,
    #     'template_domain_names': np.object,
    #     'template_sequence': np.object,
    #     'template_sum_probs': np.float32,
    # }

    all_template_features = {}
    for template_feature_name in templates.TEMPLATE_FEATURES:
        all_template_features[template_feature_name] = []

    templates_msa = []
    templates_deletion_matrix = []

    has_transforms = ('transform' in align_df.columns and
                      not any(align_df['transform'].isna()))
    cols = required_align_df_cols + optional_align_df_cols
    for _, line in align_df.iterrows():
        (template_pdbfile, target_to_template_alignstring,
         identities, target_len, template_len) = line[cols]
        if 'alt_template_sequence' in line.index:
            alt_template_sequence = line.alt_template_sequence
            identities = np.nan # too confusing
        else:
            alt_template_sequence = np.nan

        assert pd.isna(target_len) or target_len == num_res

        chains_tmp, all_resids_tmp, all_coords_tmp, all_name1s_tmp = load_pdb_coords(
            template_pdbfile, allow_chainbreaks=not args.disallow_chainbreaks,
            allow_skipped_lines = not args.disallow_skipped_lines,
        )

        crs_tmp = [(c,r) for c in chains_tmp for r in all_resids_tmp[c]]
        num_res_tmp = len(crs_tmp)
        template_full_sequence = ''.join(all_name1s_tmp[c][r] for c,r in crs_tmp)
        if not pd.isna(alt_template_sequence):
            print('using alt_template_sequence:', alt_template_sequence)
            assert len(alt_template_sequence) == len(template_full_sequence)
            template_full_sequence_orig = template_full_sequence[:]
            template_full_sequence = alt_template_sequence
        assert pd.isna(template_len) or len(template_full_sequence) == template_len

        if has_transforms:
            tl = line['transform'].split(';')
            R = np.array([float(x) for x in tl[0].split(',')]).reshape((3,3))
            v = np.array([float(x) for x in tl[1].split(',')])
            positions = [int(x) for x in tl[2].split(',')]
            print('transform: norm(R-I)', np.linalg.norm(R-np.eye(3)),
                  'norm(v)', np.linalg.norm(v),
                  'positions:', len(positions))
            for i in positions:
                c,r = crs_tmp[i]
                atoms = list(all_coords_tmp[c][r].keys())
                for atom in atoms:
                    #print('transform:', i, c, r, atom)
                    old = all_coords_tmp[c][r][atom]
                    all_coords_tmp[c][r][atom] = R @ old + v

        all_positions_tmp, all_positions_mask_tmp = fill_afold_coords(
            chains_tmp, all_resids_tmp, all_coords_tmp)

        tmp2query = {int(x.split(':')[1]) : int(x.split(':')[0]) # 0-indexed
                     for x in target_to_template_alignstring.split(';')}
        if pd.isna(identities) or args.ignore_identities:
            # note that we change the value of identities (even if it was na)
            # since we use it below for sum_probs
            identities = sum(template_full_sequence[i]==query_sequence[j]
                            for i,j in tmp2query.items())
        else:
            actual_identities = sum(template_full_sequence[i]==query_sequence[j]
                                    for i,j in tmp2query.items())
            if actual_identities != identities:
                print('actual_identities:', actual_identities, 'identities:',
                      identities, template_pdbfile,
                      target_to_template_alignstring)
                assert False, 'identities mismatch!'
                exit()

        all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
        all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num],
                                      dtype=np.int64)

        template_alseq = ['-']*num_res
        mask_atoms = ['N','CA','C','O','CB']
        #cbs = []
        for i,j in tmp2query.items(): # i=template, j=query
            template_alseq[j] = template_full_sequence[i]
            all_positions[j] = all_positions_tmp[i]
            all_positions_mask[j] = all_positions_mask_tmp[i]
            # xyz = fill_in_cbeta_coords(j, all_positions, all_positions_mask)
            # if xyz is not None:
            #     cbs.append(xyz)
            if template_full_sequence[i] in 'X-':
                # masking out this position, sequence-wise
                all_positions[j] = 0.
                all_positions_mask[j] = 0.
                for atom in mask_atoms:
                    ind = residue_constants.atom_order[atom]
                    #print('save masked coords:', i, j, template_full_sequence[i],
                    #      atom, ind)
                    all_positions[j, ind] = all_positions_tmp[i, ind]
                    all_positions_mask[j, ind] = all_positions_mask_tmp[i, ind]
                fill_in_cbeta_coords(j, all_positions, all_positions_mask)
                #print('mask total:', i, query_sequence[j],
                #      template_full_sequence_orig[i],
                #      np.sum(all_positions_mask[j]))
        #cbs = np.stack(cbs)
        #print(cbs.mean(axis=0))
        #exit()

        template_sequence = ''.join(template_alseq)
        assert len(template_sequence) == len(query_sequence) == num_res
        assert identities == sum(a==b for a,b in zip(template_sequence, query_sequence))

        template_aatype = residue_constants.sequence_to_onehot(
            template_sequence, residue_constants.HHBLITS_AA_TO_ID)

        template_pdbid = template_pdbfile.split('/')[-1][:4]

        all_template_features['template_all_atom_positions'].append(all_positions)
        all_template_features['template_all_atom_masks'].append(all_positions_mask)
        all_template_features['template_sequence'].append(template_sequence.encode())
        all_template_features['template_aatype'].append(template_aatype)
        all_template_features['template_domain_names'].append(template_pdbid.encode())
        all_template_features['template_sum_probs'].append([identities])

        query2tmp = {y:x for x,y in tmp2query.items()}
        deletion_vec = []
        alseq = ''
        last_alpos = -1
        for i in range(len(query_sequence)):
            j = query2tmp.get(i,None)
            if j is not None:
                alseq += template_full_sequence[j]
                deletion = j - (last_alpos+1)
                deletion_vec.append(deletion)
                last_alpos = j
            else:
                alseq += '-'
                deletion_vec.append(0)
        templates_msa.append(alseq)
        templates_deletion_matrix.append(deletion_vec)

        if args.verbose:
            print('identities:', identities, alignfile, template_pdbfile)
            print('Q:', query_sequence)
            print('T:', template_sequence)
            #print(alseq)
            #print(deletion_vec)
            #print(total_deletions, last_aligned_tmp, naligned)

        total_deletions = sum(deletion_vec)
        last_aligned_tmp = max(tmp2query.keys())
        naligned = len(tmp2query.keys())

        assert last_aligned_tmp+1 == naligned + total_deletions


    all_identities = np.array(all_template_features['template_sum_probs'])[:,0]
    if not args.reorder_templates:
        reorder = np.arange(all_identities.shape[0])
    else: # reorder by seqid
        assert False # we never really want to do this anymore!
        reorder = np.argsort(all_identities)[::-1]
        if args.verbose:
            print('sorted ids:', all_identities[reorder])
            print('reorder:', reorder)

    # stolen from alphafold.data.templates
    for name in all_template_features:
        all_template_features[name] = np.stack(
            [all_template_features[name][x] for x in reorder], axis=0).astype(
                templates.TEMPLATE_FEATURES[name])

    msa=[query_sequence]
    deletion_matrix=[[0]*len(query_sequence)]
    if args.add_templates_to_msa:
        msa += [templates_msa[x] for x in reorder]
        deletion_matrix += [templates_deletion_matrix[x] for x in reorder]

    assert crop_size >= len(query_sequence)

    all_metrics = run_alphafold_prediction(
        query_sequence=query_sequence,
        msa=msa,
        deletion_matrix=deletion_matrix,
        chainbreak_sequence=query_chainseq,
        template_features=all_template_features,
        model_runners=model_runners,
        out_prefix=my_outfile_prefix,
        #crop_size=crop_size,
        dump_pdbs = not (args.no_pdbs or args.terse),
        dump_metrics = not args.terse,
    )


    outl = targetl.copy()
    for model_name, metrics in all_metrics.items():
        plddts = metrics['plddt']
        paes = metrics.get('predicted_aligned_error', None)
        filetags = 'pdb plddt ptm predicted_aligned_error'.split()
        for tag in filetags:
            fname = metrics.get(tag+'file', None)
            if fname is not None:
                outl[f'{model_name}_{tag}_file'] = fname

        cs = query_chainseq.split('/')
        chain_stops = list(it.accumulate(len(x) for x in cs))
        chain_starts = [0]+chain_stops[:-1]
        nres = chain_stops[-1]
        assert nres == num_res
        outl[model_name+'_plddt'] = np.mean(plddts[:nres])
        if paes is not None:
            outl[model_name+'_pae'] = np.mean(paes[:nres,:nres])
        for chain1,(start1,stop1) in enumerate(zip(chain_starts, chain_stops)):
            outl[f'{model_name}_plddt_{chain1}'] = np.mean(plddts[start1:stop1])

            if paes is not None:
                for chain2 in range(len(cs)):
                    start2, stop2 = chain_starts[chain2], chain_stops[chain2]
                    pae = np.mean(paes[start1:stop1,start2:stop2])
                    outl[f'{model_name}_pae_{chain1}_{chain2}'] = pae
    final_dfl.append(outl)

outfile = f'{outfile_prefix}_final.tsv'
pd.DataFrame(final_dfl).to_csv(outfile, sep='\t', index=False)
print('made:', outfile)

print('total_time:', hostname, platform, timer()-start_time)
print('DONE')

exit()
