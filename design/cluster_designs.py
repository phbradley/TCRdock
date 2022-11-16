'''
'''

required_cols = ('targetid chainseq template_0_template_pdbfile model_pdbfile '
                 'template_0_target_to_template_alignstring'.split())

import argparse
parser = argparse.ArgumentParser(description="cluster loop designs")

parser.add_argument('--targets', required=True)
parser.add_argument('--outfile_prefix', required=True)
parser.add_argument('--topn', type=int, required=True)
parser.add_argument('--ranking_column', default='peptide_loop_pae')
parser.add_argument('--split_column', help='split targets based on values in this '
                    ' column and cluster separately')
parser.add_argument('--sort_descending', action='store_true')
parser.add_argument('--num_models_in_cluster_pdbfiles', type=int, default=5)
parser.add_argument('--overwrite_coords', action='store_true')

args = parser.parse_args()

import pandas as pd
targets = pd.read_table(args.targets)

for col in required_cols:
    assert col in targets.columns

assert targets.targetid.value_counts().max() == 1 # no duplicates
assert args.ranking_column in targets.columns

######################################################################################88
# more imports
import design_stats
import design_paths
design_paths.setup_import_paths()
import tcrdock as td2
from tcrdock.tcrdist.amino_acids import amino_acids
from os.path import exists
import itertools as it
import numpy as np

######################################################################################88
def gauss_cluster(D, sdev, num_clusters):
    ''' returns cluster_centers, cluster_sizes
    '''
    wts = np.array([1.0]*D.shape[0]) # how much "weight" is still available for each

    cluster_centers, cluster_sizes = [], []
    for ii in range(num_clusters):
        gauss = np.exp(-1*(D/sdev)**2) * wts[:,None] * wts[None,:]
        nbr_sum = np.sum(gauss, axis=1)
        center = np.argmax(nbr_sum)
        size = nbr_sum[center]
        member_weights = gauss[center,:]/wts[center]
        wts = np.maximum(0.0, wts - member_weights)
        assert wts[center] < 1e-3
        cluster_centers.append(center)
        cluster_sizes.append(size)

    return cluster_centers, cluster_sizes

all_tdinfo = {}
def get_tdinfo_for_pdbfile(pdbfile):
    global all_tdinfo
    if pdbfile not in all_tdinfo:
        tdifile = pdbfile+'.tcrdock_info.json'
        with open(tdifile, 'r') as f:
            tdinfo = td2.tcrdock_info.TCRdockInfo().from_string(f.read())
        all_tdinfo[pdbfile] = tdinfo
    return all_tdinfo[pdbfile]


def make_cluster_pdbfile(
        members,
        outfile,
):
    ''' by default, will re-orient MHC frame
    requires: model_pdbfile, chainseq, template_0_template_pdbfile

    needs template_0_template_pdbfile to load the tdinfo using get_tdinfo_for_pdbfile
    that's needed for defining the MHC frame
    '''
    out = open(outfile, 'w')
    for ind, l in members.reset_index().iterrows():
        pose = td2.pdblite.pose_from_pdb(l.model_pdbfile)
        tdinfo = get_tdinfo_for_pdbfile(l.template_0_template_pdbfile)
        sequence = l.chainseq.replace('/','')
        assert sequence == pose['sequence']
        assert sequence.count(l.cdr3a) == 1
        assert sequence.count(l.cdr3b) == 1
        cs = l.chainseq.split('/')
        chainbounds = [0] + list(it.accumulate(len(x) for x in cs))
        pose = td2.pdblite.set_chainbounds_and_renumber(pose, chainbounds)

        # reorient pose to have the mhc stub at the origin
        mhc_stub = td2.mhc_util.get_mhc_stub(pose, tdinfo)

        R = mhc_stub['axes'] # coordinate axes are ROW vectors in here
        v = -1* R @ mhc_stub['origin']

        pose = td2.pdblite.apply_transform_Rx_plus_v(pose, R, v)
        out.write(f'MODEL {ind+1:9d}\nREMARK {l.model_pdbfile}\n')
        td2.pdblite.dump_pdb(pose, outfile=None, out=out)
        out.write('ENDMDL\n')
    out.close()
    #print('wrote', members.shape[0], 'to file', outfile)


def run_clustering(
        coordsfile,
        #outfile_prefix,
        #topn = args.topn,
        pdbnum=args.num_models_in_cluster_pdbfiles,
):

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist, cdist, squareform

    pae_cmap_range = (2.25, 3.5)

    natoms = 4

    all_coords = np.load(coordsfile)
    info = pd.read_table(coordsfile[:-4]+'_info.tsv')
    max_cdr3len = all_coords.shape[1]//2
    num = all_coords.shape[0]
    assert all_coords.shape == (num, 2*max_cdr3len, natoms, 3)

    #for iab, ab in enumerate(['A','B','AB']):
    save_D = {}
    save_A = {}
    save_wtcdr3 = {}
    for iab, ab in enumerate(['A','B','AB']):
        if ab == 'AB':
            wt_rmsds = np.array(info['loop_rmsd'])
            wt_cdr3 = save_wtcdr3['A'] + save_wtcdr3['B']
            sdev = 1.75
            A = np.hstack([save_A['A'], save_A['B']])
            D = 0.5 * (save_D['A'] + save_D['B'])
        else: # setup, and compute D and A
            sdev = 1.0
            wt_rmsds = np.array(info[ab.lower()+'loop_rmsd'])

            info['cdr3'] = info['cdr3'+ab.lower()]

            cdr3lens = np.array(info.cdr3.str.len())
            minlen, maxlen = cdr3lens.min(), cdr3lens.max()

            wt_cdr3 = info['wt_cdr3'+ab.lower()].unique()[0]
            before = len(wt_cdr3)//2
            middle = maxlen - len(wt_cdr3)
            wt_cdr3 = wt_cdr3[:before] + '-'*middle + wt_cdr3[before:]
            save_wtcdr3[ab] = wt_cdr3

            # create coord arrays for all lens in minlen--maxlen
            # each coord array is only relevant for cdr3s with <= that length
            # for cdr3s shorter than that len, pad in the middle by repeating coords
            all_clen_coords = {}
            for clen in range(minlen, maxlen+1):
                clen_coords = []
                for ii, cdr3 in enumerate(info.cdr3):
                    my_clen = len(cdr3)
                    old_coords = all_coords[ii][iab*max_cdr3len:(iab+1)*max_cdr3len]
                    new_coords = np.zeros((clen, natoms, 3))
                    if my_clen <= clen:
                        before = my_clen//2
                        after = my_clen-before
                        middle = clen - my_clen
                        new_coords[:before] = old_coords[:before]
                        new_coords[-after:] = old_coords[before:my_clen]
                        for i in range(middle):
                            new_coords[before+i] = old_coords[before]
                    clen_coords.append(new_coords)
                all_clen_coords[clen] = np.stack(clen_coords).reshape(
                    num, clen*natoms*3)

            # now compute the distance matrix
            D = np.zeros((num,num))
            for ii in range(minlen, maxlen+1):
                ii_mask = cdr3lens==ii
                for jj in range(minlen, maxlen+1):
                    jj_mask = cdr3lens==jj
                    clen = max(ii,jj)
                    clen_coords = all_clen_coords[clen]
                    assert clen_coords.shape == (num,clen*natoms*3)
                    d = cdist(clen_coords[ii_mask], clen_coords[jj_mask])
                    d /= np.sqrt(clen*natoms)
                    assert d.shape == (ii_mask.sum(), jj_mask.sum())
                    #D[ii_mask,jj_mask] = d
                    ii_inds = np.nonzero(ii_mask)[0]
                    jj_inds = np.nonzero(jj_mask)[0]
                    D[ii_inds[:,None], jj_inds[None,:]] = d

            save_D[ab] = D

            # now fill the aa matrix
            A = np.full((num, maxlen), np.nan)
            aa2int = {x:i for i,x in enumerate(amino_acids)}
            for ii, cdr3 in enumerate(info.cdr3):
                my_clen = len(cdr3)
                before = my_clen//2
                after = my_clen-before
                aarow = np.array([aa2int[x] for x in cdr3])
                A[ii,:before] = aarow[:before]
                A[ii,-after:] = aarow[before:]
            save_A[ab] = A


        #print('hierarchical clustering')
        Z = hierarchy.linkage(squareform(D, force='tovector'),
                              method='average')
        num_clusters = 3
        centers, sizes = gauss_cluster(D, sdev, num_clusters)
        print('gauss_cluster:', minlen, maxlen, uniqtag, ab, sizes)

        Dc = D[:,centers]
        gauss = np.exp(-1*(Dc/sdev)**2)

        for c in range(num_clusters):
            inds = np.argsort(gauss[:,c])[::-1][:pdbnum] # decreasing order
            #print(gauss[:,c][inds])
            members = info.iloc[inds]
            fname = f'{coordsfile[:-4]}_{ab}_c{c}.pdb'
            make_cluster_pdbfile(members, fname)


        #print('clustermapping')
        kws = dict(cbar_kws=dict(ticks=np.arange(20)))
        #figsize=(12, 12))
        cmap = plt.get_cmap('viridis')
        row_colors = [[cmap(x) for x in gauss[:,i]]
                      for i in range(3)]
        wt_sdev = sdev
        wt_gauss = np.exp(-1*(wt_rmsds/wt_sdev)**2)
        row_colors.append([cmap(x) for x in wt_gauss])
        paes = info['peptide_loop_pae']
        mn,mx = pae_cmap_range
        print('pae range:', paes.min(), paes.max(), 'using:', mn, mx)
        cmap = plt.get_cmap('plasma_r')
        row_colors.append([cmap((x-mn)/(mx-mn)) for x in paes])

        # show wt sequence
        cmap = plt.get_cmap('tab20')
        white = (1,1,1)
        col_colors = [white if aa=='-' else cmap.colors[aa2int[aa]]
                      for aa in wt_cdr3]

        cm = sns.clustermap(
            A, row_linkage=Z, col_cluster=False, col_colors=col_colors,
            row_colors=row_colors, cmap='tab20', vmin=-0.5, vmax=19.5, **kws)
        reorder = list(cm.dendrogram_row.reordered_ind)
        cm.ax_cbar.set_yticklabels(amino_acids, fontsize=6)
        yticks, ytick_labels = [], []
        for i,c in enumerate(centers):
            pos = reorder.index(c)
            yticks.append(pos+0.5)
            ytick_labels.append(f'C{i+1}')
        cm.ax_heatmap.set_yticks(yticks)
        cm.ax_heatmap.set_yticklabels(ytick_labels)

        cm.ax_heatmap.set_xticklabels(wt_cdr3, rotation='horizontal')
        fname = f'{coordsfile[:-4]}_{ab}_cm.png'
        cm.savefig(fname)
        print('made:', fname)

mhc_cores = set() # for debugging

def make_coords_file( # and also an infofile
        targets,
        outfile,
):
    global mhc_cores
    print('make_coords_file:', targets.shape[0], outfile)
    assert len(targets.template_0_template_pdbfile.unique()==1),\
        'safer to cluster designs for different templates separately'

    atoms = [' N  ',' CA ', ' C  ', ' O  ']

    has_info = ('cdr3a' in targets.columns and 'cdr3b' in targets.columns and
                'aloop_rmsd' in targets.columns and 'loop_rmsd' in targets.columns and
                all(l.chainseq.count(l.cdr3a)==1 for l in targets.itertuples()) and
                all(l.chainseq.count(l.cdr3b)==1 for l in targets.itertuples()))
    if not has_info:
        print('adding info:', targets.shape)
        targets = design_stats.compute_stats(targets)


    tdifile = targets.template_0_template_pdbfile.unique()[0]+'.tcrdock_info.json'
    with open(tdifile, 'r') as f:
        tdinfo = td2.tcrdock_info.TCRdockInfo().from_string(f.read())

    # dimension the arrays
    # assume same max cdr3len for both a+b
    # dims: topn x 2*max_cdr3len x len(atoms) x 3 ## 4-D!
    max_cdr3len = max(max(len(x.cdr3a), len(x.cdr3b))
                      for x in targets.itertuples())
    all_coords = []
    for _, l in targets.iterrows():
        pose = td2.pdblite.pose_from_pdb(l.model_pdbfile)
        sequence = l.chainseq.replace('/','')
        assert sequence == pose['sequence']
        assert sequence.count(l.cdr3a) == 1
        assert sequence.count(l.cdr3b) == 1
        cs = l.chainseq.split('/')
        chainbounds = [0] + list(it.accumulate(len(x) for x in cs))
        pose = td2.pdblite.set_chainbounds_and_renumber(pose, chainbounds)

        # reorient pose to have the mhc stub at the origin
        mhc_core_seq = ''.join(sequence[x] for x in tdinfo.mhc_core)
        if mhc_core_seq not in mhc_cores:
            mhc_cores.add(mhc_core_seq)
            print(uniqtag, mhc_core_seq)
        mhc_stub = td2.mhc_util.get_mhc_stub(pose, tdinfo)

        R = mhc_stub['axes']
        v = -1* R @ mhc_stub['origin']

        coords = np.full((2*max_cdr3len, len(atoms), 3), np.nan)
        for ii, cdr3 in enumerate([l.cdr3a, l.cdr3b]):
            start = sequence.index(cdr3)
            for i in range(len(cdr3)):
                xyzs = pose['coords'][pose['resids'][start+i]]
                for j, atom in enumerate(atoms):
                    coords[ii*max_cdr3len + i, j] = R@xyzs[atom] + v
        all_coords.append(coords)
    all_coords = np.stack(all_coords)
    assert all_coords.shape == (all_coords.shape[0], 2*max_cdr3len, len(atoms), 3)
    np.save(outfile, all_coords)
    print('made:', outfile)
    fname2 = outfile[:-4]+'_info.tsv'
    targets.to_csv(fname2, sep = '\t', index=False)
    print('made:', fname2)


if args.split_column:
    uniqtags = sorted(set(targets[args.split_column]))
else:
    uniqtags = ['all']

targets.sort_values(args.ranking_column, ascending = not args.sort_descending,
                    inplace=True)

for uniqtag in uniqtags:
    coordsfile = f'{args.outfile_prefix}_{uniqtag}_{args.topn}_coords.npy'

    if uniqtag=='all':
        mytargets = targets.head(args.topn)
    else:
        mytargets = targets[targets[args.split_column]==uniqtag].head(args.topn)

    if exists(coordsfile) and not args.overwrite_coords:
        print('reusing old coordsfile:', coordsfile)
    else:
        make_coords_file(mytargets, coordsfile)


    run_clustering(coordsfile)
