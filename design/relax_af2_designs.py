''' Read TSV file with info on TCR:pMHC designs

* chainseq
* model_pdbfile
* template_0_target_to_template_alignstring -- defines the loop residues
 -- OR --
  template_0_alt_template_sequence

First try cartesian relaxing.

Flexibility:
- backbone: loops and peptide

'''

targets_required_cols = 'chainseq model_pdbfile'.split()

# need these in order to be able to define the flexible loops in the design
targets_need_one_of_cols = (
    'template_0_target_to_template_alignstring designable_positions').split()

import argparse

parser = argparse.ArgumentParser(
    description="Evaluate alphafold designs with Rosetta")

parser.add_argument('--targets', required=True)
parser.add_argument('--outfile_prefix', required=True)
parser.add_argument('--subset_index', type=int, default=0)
parser.add_argument('--subset_mod', type=int, default=1)
parser.add_argument('--random_delay', type=int)
parser.add_argument('--extend_flex', type=int, default=1)
parser.add_argument('--dump_pdbs', action='store_true')
parser.add_argument('--mute', action='store_true')
parser.add_argument('--ex1', action='store_true')
parser.add_argument('--ex2', action='store_true')
parser.add_argument('--norelax', action='store_true')
parser.add_argument('--beta_nov16_cart', action='store_true')
parser.add_argument('--weights_filetag', default="ref2015_cart.wts")


args = parser.parse_args()

######################################################################################88

if 'beta' in args.weights_filetag:
    assert args.beta_nov16_cart

from os.path import exists

assert exists(args.targets)

import pandas as pd

targets = pd.read_table(args.targets)
num_to_process = sum(x%args.subset_mod==args.subset_index
                     for x in targets.index)
print(f'read {targets.shape[0]} targets, will process {num_to_process}')
for col in targets_required_cols:
    assert col in targets.columns, f'Need column {col} in targets TSV file'
assert any(col in targets.columns for col in targets_need_one_of_cols)

### more imports
from timeit import default_timer as timer
import numpy as np
import sys
import pyrosetta
import itertools as it

from pyrosetta.rosetta import core, protocols, numeric, basic, utility
#from pyrosetta.rosetta.utility import vector1_bool as bools
#from pyrosetta.rosetta.numeric import xyzVector_double_t as Vector
#from pyrosetta.rosetta.utility import vector1_string as bools
#import pyrosetta.rosetta.utility.vector1_bool as bools

# local import s
from design_stats import get_designable_positions
from _superimposition_transform import superimposition_transform

# pyrosetta init
init_flags = '-ignore_unrecognized_res 1 -include_current -out:file:renumber_pdb'

if args.random_delay is not None:
    init_flags += f' -run:random_delay {args.random_delay}'
if args.mute:
    init_flags += f' -mute all'
# if not args.fast:
#     init_flags += f' -ex1 -ex2'
if args.ex1:
    init_flags += f' -ex1'
if args.ex2:
    init_flags += f' -ex2'
if args.beta_nov16_cart:
    init_flags += f' -beta_nov16_cart'

if not args.mute:
    print('init_flags:', init_flags)

pyrosetta.init(init_flags)

################################################################################
# FUNCTIONS
################################################################################


def fastrelax_pose(
        scorefxn,
        flex_positions, # bb+chi flex
        nbr_positions, # chi flex
        pose,
        nrepeats=1,
):
    # movemap:
    mm = pyrosetta.rosetta.core.kinematics.MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)
    mm.set_jump(False)

    for i in flex_positions:
        mm.set_bb(i, True)
        mm.set_chi(i, True)

    for i in nbr_positions:
        mm.set_chi(i, True)

    if not args.mute:
        print('fastrelax_peptide:: adding chi flex at',
              len([x for x in nbr_positions
                   if x not in set(flex_positions)]))

    fr = protocols.relax.FastRelax(scorefxn_in=scorefxn,
                                   standard_repeats=nrepeats)
    fr.cartesian(True)
    #fr.set_task_factory(tf)
    fr.set_movemap(mm)
    fr.set_movemap_disables_packing_of_fixed_chi_positions(True)
    # For non-Cartesian scorefunctions, use "dfpmin_armijo_nonmonotone"
    fr.min_type("lbfgs_armijo_nonmonotone")
    if not args.norelax:
        fr.apply(pose)


def unbind_tcr(peptide_chain, pose, sep=50):
    from pyrosetta.rosetta.numeric import xyzVector_double_t as Vector
    from pyrosetta.rosetta.numeric import xyzMatrix_double_t as Matrix
    posl = range(1,pose.size()+1)

    chains = np.array([pose.chain(x) for x in posl])
    calphas = np.array([pose.residue(x).xyz("CA") for x in posl])

    mhc_cen = np.mean(calphas[chains<peptide_chain], axis=0)
    tcr_cen = np.mean(calphas[chains>peptide_chain], axis=0)

    trans = tcr_cen - mhc_cen
    trans = Vector(*trans).normalized()
    trans *= sep

    pose2 = pose.clone()
    rotation = Matrix.I()
    pose2.apply_transform_Rx_plus_v(rotation, trans)

    for pos in range(pose.chain_begin(peptide_chain+1), pose.size()+1):
        pose.replace_residue(pos, pose2.residue(pos), False)




def find_neighbors(
        core_positions,
        pose,
        heavyatom_distance_threshold = 6.0,
):
    nbr_positions = set(core_positions)

    posl = range(1, pose.size()+1)

    for i in posl:
        rsd1 = pose.residue(i)
        if rsd1.is_virtual_residue():
            continue
        for j in core_positions:
            rsd2 = pose.residue(j)
            dis2 = rsd1.nbr_atom_xyz().distance_squared(rsd2.nbr_atom_xyz())
            threshold = (rsd1.nbr_radius() + rsd2.nbr_radius() +
                         heavyatom_distance_threshold)**2
            if dis2 <= threshold:
                nbr_positions.add(i)
                break
    return nbr_positions



def superimpose_on_calphas(
        fix_pose,
        mov_pose,
        debug_rmsd=False,
):
    from pyrosetta.rosetta.numeric import xyzVector_double_t as Vector
    from pyrosetta.rosetta.numeric import xyzMatrix_double_t as Matrix
    from pyrosetta.rosetta.utility import vector1_numeric_xyzVector_double_t as Vectors

    fixcoords = Vectors()
    movcoords = Vectors()
    nres = fix_pose.size()
    assert mov_pose.size() == nres

    for i in range(1,nres+1):
        movcoords.append(mov_pose.residue(i).xyz("CA"));
        fixcoords.append(fix_pose.residue(i).xyz("CA"));

    if debug_rmsd:
        # no side effect on coords
        rmsd = numeric.model_quality.calc_rms(fixcoords, movcoords)

    # this has a side effect of translating both COMs to the origin
    rotation, translation = superimposition_transform(fixcoords, movcoords)

    mov_pose.apply_transform_Rx_plus_v(rotation, translation)

    if debug_rmsd:
        rmsd_redo=0.0
        for i in range(1,nres+1):
            rmsd_redo += mov_pose.residue(i).xyz("CA").distance_squared(
                fix_pose.residue(i).xyz("CA"))
        rmsd_redo = np.sqrt(rmsd_redo/len(fixcoords))
        print('rmsd:', rmsd, 'rmsd_redo:', rmsd_redo)


def read_alphafold_pose(filename):
    pose = pyrosetta.pose_from_pdb(filename)

    pdbinfo = pose.pdb_info()

    # insert chainbreaks at residue numbering breaks
    for i in range(1,pose.size()):
        if pdbinfo.number(i+1) != pdbinfo.number(i)+1:
            pose.conformation().insert_chain_ending(i)
            core.pose.add_upper_terminus_type_to_pose_residue(pose, i)
            core.pose.add_lower_terminus_type_to_pose_residue(pose, i+1)

    return pose


def delete_positions_and_rescore(
        scorefxn,
        posl_in,
        pose_in,
):
    ''' We should be more careful about ramachandran and paa and dunbrack scoring
    at "fake" termini that are created by deleting bonded residues...
    The phi/psi angles assigned to those positions are probably pretty wonky

    We could add chainbreak variants, maybe....

    '''
    pose = pose_in.clone()

    # delete contiguous stretches
    posl = sorted(posl_in)

    while posl:
        # delete contiguous stretch
        stop = posl[-1]
        start = posl[-1]
        del posl[-1]
        while posl and posl[-1]+1==start:
            start = posl[-1]
            del posl[-1]
        pose.conformation().delete_residue_range_slow(start, stop)
    score = scorefxn(pose)
    return score




################################################################################
################################################################################


out_tsvfile = args.outfile_prefix+'_relax_af2_designs.tsv'

scorefxn = pyrosetta.create_score_function(args.weights_filetag)
eval_scorefxn = pyrosetta.create_score_function(args.weights_filetag)
eval_scorefxn.set_weight(core.scoring.cart_bonded, 0.)


dfl = []

for ind, row in targets.iterrows():
    if args.subset_mod and ind%args.subset_mod != args.subset_index:
        continue

    pose = read_alphafold_pose(row.model_pdbfile)
    cs = row.chainseq.split('/')
    cbs = [0] + list(it.accumulate(len(x) for x in cs))
    assert len(cs) == pose.num_chains()
    assert pose.num_chains() in [4,5]

    # this will likely be everything in the CDR3s between CAX and XF
    loop_posl = [x+1 for x in get_designable_positions(row=row)]

    nres_mhc, nres_pmhc = cbs[-4:-2]
    peptide_chain = pose.num_chains()-2
    pep_posl = list(range(nres_mhc+1, nres_pmhc+1)) # 1-indexed

    start_pose = pose.clone()

    flex_positions = pep_posl + loop_posl
    nbr_positions = find_neighbors(flex_positions, pose)

    start_time = timer()
    fastrelax_pose(scorefxn, flex_positions, nbr_positions, pose)
    relax_time = timer() - start_time
    if args.dump_pdbs:
        # if hasattr(row, 'targetid'):
        #     basename = row.targetid.replace('/','_')
        # else:
        basename = row.model_pdbfile.split('/')[-1][:-4]
        pdb_fname = args.outfile_prefix+'_'+basename+'_relax.pdb'
        pose.dump_pdb(pdb_fname)
        print('made:', pdb_fname)

    start_time = timer()
    bound_score = eval_scorefxn(pose)

    # calculate interaction energies between peptide and loops
    unbound_pose = pose.clone()

    unbind_tcr(peptide_chain, unbound_pose)
    unbound_score_frozen = eval_scorefxn(unbound_pose)

    binding_energy_frozen = bound_score - unbound_score_frozen

    # delete stuff
    no_pep_score = delete_positions_and_rescore(eval_scorefxn, pep_posl, pose)
    no_loop_score = delete_positions_and_rescore(eval_scorefxn, loop_posl, pose)
    no_pep_loop_score = delete_positions_and_rescore(
        eval_scorefxn, pep_posl+loop_posl, pose)


    pep_score = bound_score - no_pep_score
    loop_score = bound_score - no_loop_score
    pep_loop_score = bound_score - no_pep_loop_score
    pep_loop_intxn = pep_score + loop_score - pep_loop_score

    # compute rmsds over flexible regions
    superimpose_on_calphas(start_pose, pose)
    pep_rmsd=0.
    rmsd_atoms = 'N CA C O'.split()
    for pos in pep_posl:
        for atom in rmsd_atoms:
            pep_rmsd += start_pose.residue(pos).xyz(atom).distance_squared(
                pose.residue(pos).xyz(atom))
    pep_rmsd /= len(pep_posl)*len(rmsd_atoms)
    loop_rmsd=0.
    for pos in loop_posl:
        for atom in rmsd_atoms:
            loop_rmsd += start_pose.residue(pos).xyz(atom).distance_squared(
                pose.residue(pos).xyz(atom))
    loop_rmsd /= len(loop_posl)*len(rmsd_atoms)
    eval_time = timer() - start_time

    # save final results
    outl = row.copy()
    sequence = pose.sequence()
    updates = dict(
        bound_score=bound_score,
        unbound_score_frozen=unbound_score_frozen,
        binding_energy_frozen=binding_energy_frozen,
        no_pep_score=no_pep_score,
        no_loop_score=no_loop_score,
        no_pep_loop_score=no_pep_loop_score,
        pep_score = pep_score,
        loop_score = loop_score,
        pep_loop_score = pep_loop_score,
        pep_loop_intxn = pep_loop_intxn,
        relaxed_peptide_rmsd = pep_rmsd,
        relaxed_loop_rmsd = loop_rmsd,
        nflex_peptide = len(pep_posl),
        seq_peptide = ''.join(sequence[x-1] for x in pep_posl),
        nflex_loop = len(loop_posl),
        seq_loop = ''.join(sequence[x-1] for x in loop_posl),
        relax_time = relax_time,
        eval_time = eval_time,
    )
    for k,v in updates.items():
        outl[k] = v
    if args.dump_pdbs:
        outl['relaxed_model_pdbfile'] = pdb_fname
    dfl.append(outl)

    pd.DataFrame(dfl).to_csv(out_tsvfile, sep='\t', index=False)
    print('relax_time:', relax_time, 'eval_time:', eval_time)

pd.DataFrame(dfl).to_csv(out_tsvfile, sep='\t', index=False)


print('DONE')
