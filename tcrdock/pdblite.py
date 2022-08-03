from .util import long2short
from . import tcrdist
import numpy as np
import itertools as it
from sys import exit
from collections import Counter, OrderedDict
import copy

long2short_MSE = dict(**long2short, MSE='M')

def load_pdb_coords(
        pdbfile,
        allow_chainbreaks=False,
        allow_skipped_lines=False,
        verbose=False,
        preserve_atom_name_whitespace=False,
        require_CA=False,
        require_bb=False,
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
            if line[:6] == 'ENDMDL':
                #print('stopping ENDMDL:', pdbfile)
                break
            if (line[:6] in ['ATOM  ','HETATM'] and line[17:20] != 'HOH' and
                line[16] in ' A1'):
                if line[17:20] in long2short_MSE:
                    resid = line[22:27]
                    chain = line[21]
                    if chain not in all_resids:
                        all_resids[chain] = []
                        all_coords[chain] = {}
                        all_name1s[chain] = {}
                        chains.append(chain)
                    if line.startswith('HETATM') and line[12:16] == ' CA ':
                        print('WARNING: HETATM', pdbfile, line[:-1])
                    if preserve_atom_name_whitespace:
                        atom = line[12:16]
                    else:
                        atom = line[12:16].strip()
                    if resid not in all_resids[chain]:
                        all_resids[chain].append(resid)
                        all_coords[chain][resid] = OrderedDict()
                        all_name1s[chain][resid] = long2short_MSE[line[17:20]]

                    all_coords[chain][resid][atom] = np.array(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])])
                else:
                    if verbose or line[12:16] == ' CA ':
                        print('skip ATOM line:', line[:-1], pdbfile)
                    skipped_lines = True

    # possibly subset to residues with CA
    N, CA, C  = ' N  ', ' CA ', ' C  '
    require_atoms = [N,CA,C] if require_bb else [CA] if require_CA else []
    if require_atoms:
        if not preserve_atom_name_whitespace:
            require_atoms = [x.strip() for x in require_atoms]
        chains = all_resids.keys()
        for chain in chains:
            bad_resids = [x for x,y in all_coords[chain].items()
                          if any(a not in y for a in require_atoms)]
            if bad_resids:
                print('missing one of', require_atoms, bad_resids)
                for r in bad_resids:
                    all_resids[chain].remove(r)
                    del all_coords[chain][r]
                    del all_name1s[chain][r]


    # check for chainbreaks
    maxdis = 1.75
    for chain in chains:
        for res1, res2 in zip(all_resids[chain][:-1], all_resids[chain][1:]):
            coords1 = all_coords[chain][res1]
            coords2 = all_coords[chain][res2]
            if 'C' in coords1 and 'N' in coords2:
                dis = np.sqrt(np.sum(np.square(coords1['C']-coords2['N'])))
                if dis>maxdis:
                    if verbose or not allow_chainbreaks:
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

def load_pdb_coords_resids(
        pdbfile,
        **kwargs,
        #allow_chainbreaks=False,
        #allow_skipped_lines=False,
        #verbose=False,
):
    ''' returns: resids, coords, sequence

    resids is a list of (chain, resid) tuples

    coords is a dict indexed by (chain, resid)

    sequence is the full sequence, as a string
    '''

    chains, all_resids, all_coords, all_name1s = load_pdb_coords(
        pdbfile, **kwargs)#allow_chainbreaks, allow_skipped_lines, verbose)

    resids = list(it.chain(*[[(c,r) for r in all_resids[c]]
                               for c in chains]))
    coords = {(c,r):all_coords[c][r]
              for c in chains
              for r in all_resids[c]}

    sequence = ''.join(all_name1s[c][r] for c,r in resids)

    return resids, coords, sequence


def pose_from_pdb(filename, **kwargs):
    ''' calls update_derived_data(pose) before returning pose
    '''
    defaults = dict(
        allow_chainbreaks=True,
        allow_skipped_lines=True,
        preserve_atom_name_whitespace=True,
        require_CA=True,
        require_bb=True,
    )
    kwargs = {**defaults, **kwargs}
    resids, coords, sequence = load_pdb_coords_resids(filename, **kwargs)
    pose = {'resids':resids, 'coords':coords, 'sequence':sequence}
    pose = update_derived_data(pose)

    return pose


def save_pdb_coords(outfile, resids, coords, sequence, verbose=False, bfactors=None):
    ''' right now bfactors is a list of length = resids (all atoms in res have same)
    '''
    assert len(sequence) == len(resids)
    out = open(outfile, 'w')
    last_chain = None
    counter=0
    if bfactors is None:
        bfactors = it.repeat(50.0)
    else:
        assert len(bfactors) == len(resids)
    for cr, name1, bfac in zip(resids, sequence, bfactors):
        (chain,resid) = cr
        name3 = tcrdist.amino_acids.short_to_long[name1]
        if chain != last_chain and last_chain is not None:
            out.write('TER\n')
        last_chain = chain
        for atom,xyz in coords[cr].items():
            counter += 1
            a = atom.strip()
            assert len(atom) == 4
            assert len(resid) == 5
            element = 'H' if a[0].isdigit() else a[0]
            occ=1.
            #                   6:12      12:16   17:20    21    22:27
            outline = (f'ATOM  {counter:6d}{atom} {name3} {chain}{resid}   '
                       f'{xyz[0]:8.2f}{xyz[1]:8.2f}{xyz[2]:8.2f}{occ:6.2f}{bfac:6.2f}'
                       f'{element:>12s}\n')
            assert (outline[12:16] == atom and outline[17:20] == name3 and
                    outline[22:27] == resid) # confirm register
            out.write(outline)
    out.close()
    if verbose:
        print('made:', outfile)


def dump_pdb(pose, outfile):
    save_pdb_coords(outfile, pose['resids'], pose['coords'], pose['sequence'])


def check_coords_shape(pose):
    coords = pose['coords']
    for r in pose['resids']:
        for a,xyz in coords[r].items():
            if xyz.shape != (3,):
                print('check_coords_shape: FAIL:', r, a, xyz)
                assert False

def update_derived_data(pose):
    ''' pose is a python dict with keys

    'resids' - list of (chain,resid) tuples where resid is the pdb resid: line[22:27]
    'coords' - dict indexed by (chain,resid) tuples mapping to dicts from atom-->xyz
    'sequence' - string, sequence in 1-letter code

    invariants:

    len(resids) == len(sequence)
    set(coords.keys()) == set(resids)
    each chain id occurs in contiguous block of resids
    each (chain,resid) has N,CA,C, names = ' N  ', ' CA ', ' C  '

    sets up derived data:

    'ca_coords'
    'chains'
    'chainseq'
    'chainbounds'

    '''
    #check_coords_shape(pose)

    CA = ' CA '

    resids, coords, sequence = pose['resids'], pose['coords'], pose['sequence']

    chains = [x[0] for x in it.groupby(resids, lambda x:x[0])]
    assert len(set(chains)) == len(chains) # not interspersed: each chain comes once

    chainseq = {c:'' for c in chains}
    for r,a in zip(resids, sequence):
        chainseq[r[0]] += a

    chain_lens = [len(chainseq[c]) for c in chains]
    chainbounds = [0] + list(it.accumulate(chain_lens))

    chainseq = '/'.join(chainseq[c] for c in chains)

    assert len(chainbounds) == len(chains) + 1 # 0 at the beginning, N at the end
    pose['ca_coords'] = np.stack([coords[r][CA] for r in resids])
    pose['chains'] = chains
    pose['chainseq'] = chainseq
    pose['chainbounds'] = chainbounds

    return pose

def renumber(pose):
    ''' set resids from 0 ---> N-1
    set chains from A,B,C,...Z
    '''

    resids, coords = pose['resids'], pose['coords']

    old_chains = pose['chains']
    new_chains = [chr(ord('A')+i) for i in range(len(old_chains))]

    old2new = dict(zip(old_chains, new_chains))

    new_resids = []
    new_coords = {}

    for ii,r in enumerate(resids):
        new_r = (old2new[r[0]], f'{ii:4d} ')
        new_resids.append(new_r)
        new_coords[new_r] = copy.deepcopy(coords[r])

    pose['resids'] = new_resids
    pose['coords'] = new_coords

    return update_derived_data(pose)

def set_chainbounds_and_renumber(pose, chainbounds):
    ''' set resids from 0 ---> N-1
    set chains from A,B,C,...Z
    '''

    assert chainbounds[0] == 0 and chainbounds[-1] == len(pose['sequence'])

    resids, coords = pose['resids'], pose['coords']

    num_chains = len(chainbounds)-1
    new_chains = [chr(ord('A')+i) for i in range(num_chains)]

    new_resids = []
    new_coords = {}

    for c, chain in enumerate(new_chains):
        assert chainbounds[c] < chainbounds[c+1]
        for ind in range(chainbounds[c], chainbounds[c+1]):
            new_r = (chain, f'{ind:4d} ')
            r = resids[ind]
            new_resids.append(new_r)
            new_coords[new_r] = copy.deepcopy(coords[r])

    assert len(new_resids) == len(resids) == len(new_coords.keys())

    pose['resids'] = new_resids
    pose['coords'] = new_coords

    return update_derived_data(pose)


def apply_transform_Rx_plus_v(pose, R, v):
    assert R.shape==(3,3) and v.shape==(3,)
    resids, coords = pose['resids'], pose['coords']
    for r in resids:
        coords[r] = {a:R@xyz + v for a,xyz in coords[r].items()}

    pose['coords'] = coords # probably not necessary

    # update since coords changed
    return update_derived_data(pose)

def delete_chains(pose, chain_nums):
    '''
    '''
    del_chains = [pose['chains'][c] for c in chain_nums]

    resids, coords, sequence = pose['resids'], pose['coords'], pose['sequence']

    new_resids = []
    new_sequence = []
    for r,a in zip(resids, sequence):
        if r[0] in del_chains:
            del coords[r]
        else:
            new_resids.append(r)
            new_sequence.append(a)

    pose['resids'] = new_resids
    pose['sequence'] = ''.join(new_sequence)
    pose['coords'] = coords

    return update_derived_data(pose)

def append_chains(pose, src_pose, src_chain_nums):
    assert pose is not src_pose


    resids, coords = pose['resids'], pose['coords']
    src_resids, src_coords, src_sequence = (src_pose['resids'], src_pose['coords'],
                                            src_pose['sequence'])

    ord0 = ord(max(pose['chains']))+1

    sequence = list(pose['sequence'])
    for ii, chain_num in enumerate(src_chain_nums):
        old_chain = src_pose['chains'][chain_num]
        new_chain = chr(ord0+ii)
        for r,a in zip(src_resids, src_sequence):
            if r[0] == old_chain:
                new_r = (new_chain, r[1])
                resids.append(new_r)
                sequence.append(a)
                coords[new_r] = copy.deepcopy(src_coords[r])
    pose['resids'] = resids
    pose['coords'] = coords
    pose['sequence'] = ''.join(sequence)

    return update_derived_data(pose)



