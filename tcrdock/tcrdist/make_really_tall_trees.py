######################################################################################88
from .basic import *
from . import score_trees_devel
from .tcrdist_svg_basic import (
    make_text, SVG_tree_plotter, rgb_from_fraction, create_file)
from . import util
from . import html_colors
from . import tcr_sampler
import scipy.stats
import numpy as np
import pandas as pd

#from functools import reduce
#from mannwhitneyu import mannwhitneyu as mannwhitneyu_exact #too slow

#font_family = 'monospace'
#font_family = 'Droid Sans Mono'
#font_family = 'DejaVu Sans Mono'
font_family = 'courier'

gap_character = '-' ## different from some other places
min_cluster_size = 1


#tree_width = 750 #hack=450
#ymargin = 30 ## right now we dont want top text to get cut off
#xmargin = 10

text_column_separation = 4
labels_tree_separation = 10

#branch_width_fraction = 0.1 # making argument of make_tall_tcr_tree

def _pad_to_middle( s, num ): ## with spaces
    if len(s) >= num:return s
    extra = num-len(s)
    before = extra//2
    after = extra-before
    return ' '*before + s + ' '*after

def _get_primary_number( gene_name ):
    tmp = gene_name[:]
    while tmp and not tmp[0].isdigit():
        tmp = tmp[1:]
    if not tmp:
        ## for example, if gene_name=='TRGJP'
        return 0
    assert tmp[0].isdigit()
    if tmp.isdigit():
        return int(tmp)
    else:
        tmp2 = ''
        while tmp[0].isdigit():
            tmp2 += tmp[0]
            tmp = tmp[1:]
        return int(tmp2)

def _get_primary_number_jb(gene_name):
    g = gene_name[4:gene_name.index('*')]
    if len(g) == 3 and g[1] == '-' and (g[0]+g[2]).isdigit():
        return int(g[0]+g[2])
    else:
        return _get_primary_number(gene_name)


def make_tall_tcr_tree(
        tcrs,
        organism,
        tcrdist_matrix, # symmetric numpy array
        pngfile,
        tcrdist_clustering_threshold=77.1, # not really important, sets tree order
        color_scores=None,
        color_score_range=None,
        dont_trim_labels=False,
        clone_sizes=None,
        extra_tcr_labels_df=None,
        extra_tcr_labels_to_color_by_color_score=[],
        ignore_nucseqs=False, # if True, don't need cdr3a/b_nucseq in tcrs
        verbose=False,
        cmap=None,
        node_score_func=None, # takes list of leaves, returns node score
        force_max_tcrdist=None,
        branch_width_fraction = 0.1,
        top_margin = 30,
        bottom_margin = 30,
        left_margin = 10,
        right_margin = 10,
        tree_width = 750,
        label_fontsize = 10, # tree_height = len(tcrs)*label_fontsize
        skip_text = False,
        skip_rmsd_bar = False,
        return_leaf_indices = False,
):
    ''' tcrs is a list of tuples: tcrs = [ (atcr1,btcr1), (atcr2,btcr2), ....
    atcr1 = (va1, ja1, cdr3a1, cdr3a_nucseq1, *)
    btcr1 = (vb1, jb1, cdr3b1, cdr3b_nucseq1, *)

    minimally tested, currently under development
    '''
    N= len(tcrs)
    assert tcrdist_matrix.shape == (N,N)

    ab = 'AB' # paired trees, for starters

    if color_scores is None:
        color_scores = [1.0]*N
    if color_score_range is None:
        color_score_range = (min(color_scores), max(color_scores))
        if color_score_range[0] == color_score_range[1]:
            color_score_range = (color_score_range[0], color_score_range[0]+.1)
    #if clone_sizes is None:
    #    clone_sizes = [1]*N

    if not ignore_nucseqs:
        if verbose:
            print('parsing junctions for', len(tcrs), 'tcrs')
        junctions = tcr_sampler.parse_tcr_junctions(organism, tcrs)
    else:
        junctions = pd.DataFrame(dict(
            va   =[x[0][0] for x in tcrs],
            ja   =[x[0][1] for x in tcrs],
            cdr3a=[x[0][2] for x in tcrs],
            vb   =[x[1][0] for x in tcrs],
            jb   =[x[1][1] for x in tcrs],
            cdr3b=[x[1][2] for x in tcrs],
        ))

    va_colors, ja_colors, vb_colors, jb_colors = util.assign_colors_to_conga_tcrs(
        tcrs, organism)

    junctions['va_color'] = va_colors
    junctions['ja_color'] = ja_colors
    junctions['vb_color'] = vb_colors
    junctions['jb_color'] = jb_colors

    junctions['color_score'] = color_scores
    if clone_sizes is not None:
        junctions['clone_size'] = list(clone_sizes)

    if extra_tcr_labels_df is not None:
        assert extra_tcr_labels_df.shape[0] == N
        extra_tcr_labels = list(extra_tcr_labels_df.columns)
        for label in extra_tcr_labels:
            junctions[label] = list(extra_tcr_labels_df[label])
    else:
        extra_tcr_labels = []



    max_clone_count = None if clone_sizes is None else np.max(clone_sizes)

    def clonality_fraction( clone_size ):
        #global max_clone_count
        if max_clone_count==1: return 0.0
        exponent = 1.0/3
        mx = max_clone_count**exponent
        cs = clone_size**exponent
        return ( cs-1.0)/(mx-1.0)

    def clonality_color( clone_size ):
        return rgb_from_fraction( clonality_fraction( clone_size ) )


    all_nbrs = []
    for dists in tcrdist_matrix:
        nbrs = []
        for ii,d in enumerate(dists):
            if d <= tcrdist_clustering_threshold:
                nbrs.append( ii )
        all_nbrs.append( nbrs )

    deleted = [False]*N

    centers = []
    all_members = []

    if verbose:
        print(f'clustering {len(tcrs)} tcrs')
    while True:
        clusterno = len(centers)

        best_nbr_count =0
        for i in range(N):
            if deleted[i]: continue
            nbr_count = 0
            for nbr in all_nbrs[i]:
                if not deleted[nbr]:
                    nbr_count+=1
            if nbr_count > best_nbr_count:
                best_nbr_count = nbr_count
                center = i

        if best_nbr_count < min_cluster_size:
            break

        centers.append( center )
        members = [center]

        deleted[center] = True
        for nbr in all_nbrs[center]:
            if not deleted[nbr]:
                deleted[nbr] = True
                members.append( nbr )

        assert len(members) == best_nbr_count
        all_members.append( frozenset(members) )

    num_clusters = len(centers)
    num_tcrs = len(tcrs)

    ## I think this will give a better ordering of the TCRs along the tree
    ## order from 1....N by going through the members of the clusters,
    ##   largest to smallest
    old2new_index = {}
    new2old_index = {}
    last_index=-1
    for members in all_members:
        for member in members:
            last_index += 1
            old2new_index[ member ] = last_index
            new2old_index[ last_index ] = member
    assert len(old2new_index) == num_tcrs

    ## how much vertical space will we need?
    ##
    #label_fontsize = 10 #hack=3
    tree_height = label_fontsize * len(tcrs)
    total_svg_height = tree_height + top_margin + bottom_margin

    ## now get some info together for plotting
    all_center_dists = {}
    all_scores = []
    sizes = []
    names = []
    infos = [] ## for the color score correlations

    for new_index in range(num_tcrs):
        old_index = new2old_index[ new_index ]
        tcr = junctions.iloc[old_index]
        names.append('')
        sizes.append(1)
        infos.append(f'{tcr.cdr3a[3:-2]} {tcr.cdr3b[3:-2]}')
        all_scores.append( [tcr.color_score] )
        for other_new_index in range(num_tcrs):
            other_old_index = new2old_index[ other_new_index ]
            dist = tcrdist_matrix[ old_index ][ other_old_index ]
            all_center_dists[ (new_index,other_new_index) ] = dist
            all_center_dists[ (other_new_index,new_index) ] = dist

    percentile = -1 # means average

    if node_score_func is None:
        node_scorer = score_trees_devel.CallAverageScore(percentile)
    else:
        node_scorer = (lambda leaves,scores :
                       node_score_func([new2old_index[x] for x in leaves]))

    tree = score_trees_devel.Make_tree_new(
        all_center_dists, len(names), score_trees_devel.Update_distance_matrix_AL,
        all_scores, node_scorer,
    )

    ## look for branches with high/low scores
    #edge_pvals = {}
    #tree_splits_ttest(edge_pvals, tree, [], all_scores, infos,
    #                  epitope+'_'+ab+"_"+color_scheme )


    ## the x-values dont matter here
    ## but the y-values do
    tree_p0 = [10, top_margin ]
    tree_p1 = [1000, tree_height+top_margin ]


    ## node_position tells us where the different clusters are located, vertically
    ##
    tmp_plotter = SVG_tree_plotter()
    node_position,Transform,canvas_tree_min_rmsd, canvas_tree_w_factor \
        = score_trees_devel.Canvas_tree(
            tree, names, sizes, tree_p0, tree_p1, branch_width_fraction,
            tmp_plotter, label_internal_nodes = False,
            score_range_for_coloring = color_score_range )

    cmds = []
    # cmds.append(
    #     make_text(f'#tcrs: {len(tcrs)}', (10,ymargin), 30, font_family=font_family))

    ## now let's add some text for each tcr
    num_columns = 1 + 2*len(ab) + len(extra_tcr_labels) # cdrs + v/j * a/b + extra
    if clone_sizes is not None:
        num_columns += 1
    text_columns = []
    for i in range( num_columns ):
        text_columns.append( [] )
    header = ['']*num_columns

    #for old_index, tcr in enumerate( tcrs ):
    for old_index, tcr in junctions.iterrows():
        ## 1. cdr3s, indels
        ## 2. clonality text
        ## 3. gene segments text
        ##
        icol = 0

        text0 = ''
        htext = ''
        CDR_START,CDR_STOP = (3,-2) if not dont_trim_labels else (0,None)
        CDR_W = 18 if dont_trim_labels else 15
        if 'A' in ab:
            if ignore_nucseqs:
                text0 += _pad_to_middle(tcr.cdr3a[CDR_START:CDR_STOP], CDR_W)
                htext += _pad_to_middle('CDR3A', CDR_W)
            else:
                text0 += '{} {} {}'.format(
                    _pad_to_middle(tcr.cdr3a[CDR_START:CDR_STOP], CDR_W),
                    _pad_to_middle(tcr.cdr3a_protseq_masked[CDR_START:CDR_STOP], CDR_W),
                    _pad_to_middle(tcr.a_indels, 6))
                htext += '  cdr3a,cdr3a_masked,a_indels  '
        if 'B' in ab:
            if text0:
                text0 += ' '
            if ignore_nucseqs:
                text0 += _pad_to_middle(tcr.cdr3b[CDR_START:CDR_STOP], CDR_W)
                htext += _pad_to_middle('CDR3B', CDR_W)
            else:
                text0 += '{} {} {}'.format(
                    _pad_to_middle(tcr.cdr3b[CDR_START:CDR_STOP], CDR_W),
                    _pad_to_middle(tcr.cdr3b_protseq_masked[CDR_START:CDR_STOP], CDR_W),
                    _pad_to_middle(tcr.b_indels, 6))
                htext += '  cdr3b,cdr3b_masked,b_indels  '

        text_columns[ icol ].append( ( text0, 'black' ) )
        header[icol] = htext
        icol += 1

        if clone_sizes is not None:
            ## clonality
            header[icol] = '  C'
            text_columns[icol].append((f'{tcr.clone_size:3d}',
                                       clonality_color(tcr.clone_size)))
            icol += 1

        ## gene segments
        if 'A' in ab:
            text_columns[icol].append(
                (f'{tcr.va[2:4]}{_get_primary_number(tcr.va):02d}', tcr.va_color))
            header[icol] = 'TRAV'
            icol += 1
            text_columns[icol].append(
                (f'{tcr.ja[2:4]}{_get_primary_number(tcr.ja):02d}', tcr.ja_color))
            header[icol] = 'TRAJ'
            icol += 1
        if 'B' in ab:
            text_columns[icol].append(
                (f'{tcr.vb[2:4]}{_get_primary_number(tcr.vb):02d}', tcr.vb_color))
            header[icol] = 'TRBV'
            icol += 1
            text_columns[icol].append(
                (f'{tcr.jb[2:4]}{_get_primary_number_jb(tcr.jb):02d}', tcr.jb_color))
            header[icol] = 'TRBJ'
            icol += 1

        for label in extra_tcr_labels:
            if label in extra_tcr_labels_to_color_by_color_score:
                mn,mx = color_score_range
                frac = max(0, min(1, (tcr.color_score-mn)/(mx-mn)))
                color = rgb_from_fraction(frac, cmap=cmap)
            else:
                color = 'black'
            text_columns[icol].append((tcr[label], color))
            header[icol] = label
            icol += 1


    ## now go through and figure out how wide each of the text columns is
    x_offset = left_margin
    for col,header_tag in zip( text_columns, header ):
        assert len(col) == num_tcrs
        maxlen = max((len(x[0]) for x in col ))
        if not maxlen: continue
        for old_index, ( text,color ) in enumerate(col):
            new_index = old2new_index[ old_index ]
            ypos = node_position[ new_index ]
            lower_left = [ x_offset, ypos+0.5*label_fontsize*0.75 ]
            cmds.append(
                make_text(text, lower_left, label_fontsize, color=color,
                          font_family=font_family))
        if header_tag:
            max_ypos = max( node_position.values() )
            lower_left = [ x_offset, max_ypos+2.0*label_fontsize*0.75 ]
            htag = header_tag if len(header_tag)<=maxlen else header_tag[:maxlen-1]+'-'
            cmds.append(
                make_text(htag, lower_left, label_fontsize, color='black',
                          font_family=font_family))

        x_offset += text_column_separation + label_fontsize * 0.6 * maxlen

    if skip_text:
        cmds = []
        x_offset = left_margin  - labels_tree_separation

    ## how wide should the tree be?

    tree_p0 = [x_offset + labels_tree_separation, top_margin ]
    tree_p1 = [tree_p0[0] + tree_width, tree_height+top_margin ]

    total_svg_width = tree_p1[0] + right_margin

    ## node_position tells us where the different clusters are located, vertically
    ##

    plotter = SVG_tree_plotter(cmap=cmap)
    node_position,Transform,canvas_tree_min_rmsd, canvas_tree_w_factor = \
        score_trees_devel.Canvas_tree(
            tree, names, sizes, tree_p0, tree_p1, branch_width_fraction,
            plotter, label_internal_nodes = False,
            rmsd_bar_label_stepsize=50, force_min_rmsd=0,
            score_range_for_coloring = color_score_range,
            force_max_rmsd=force_max_tcrdist,
            show_rmsd_bar = not skip_text,
        )

    cmds.extend( plotter.cmds )

    #print(node_position)
    ## label pvals
    # if edge_pvals:
    #     plotting_info=(sizes, node_position, Transform,
    #                    canvas_tree_w_factor, canvas_tree_min_rmsd)
    #     label_pval_edges( cmds, edge_pvals, tree, plotting_info )

    create_file(cmds, total_svg_width, total_svg_height, pngfile[:-3]+'svg',
                create_png=True)

    if verbose:
        print('made:', pngfile)
        print('made:', pngfile[:-3]+'svg')


    if return_leaf_indices:
        leaves = [y for _,y in sorted(
            [(node_position[old2new_index[x]], x) for x in range(len(tcrs))])]
        return leaves

