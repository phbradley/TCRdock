######################################################################################88
from collections import OrderedDict, Counter
import os
import sys
from os.path import exists
from .tcrdist.all_genes import all_genes
from .tcrdist.amino_acids import amino_acids
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
from Bio.pairwise2 import format_alignment
from .util import path_to_db
from . import docking_geometry
from .docking_geometry import DockingGeometry
from .tcrdock_info import TCRdockInfo
from . import tcrdist
from . import superimpose
from . import tcr_util
from . import pdblite
import pandas as pd
import numpy as np
import random
import copy
from numpy.linalg import norm

CLASS2_PEPLEN = 1+9+1
TCR_CORE_LEN = 13
ALL_GENES_GAP_CHAR = '.'


# human only
human_structure_alignments = pd.read_table(path_to_db/'new_human_vg_alignments_v1.tsv')
human_structure_alignments.set_index('v_gene', drop=True, inplace=True)

# human and mouse
both_structure_alignments = pd.read_table(path_to_db/'new_both_vg_alignments_v1.tsv')
both_structure_alignments.set_index(['organism','v_gene'], drop=True, inplace=True)

def read_fasta(filename): # helper
    ''' return OrderedDict indexed by the ">" lines (everything after >)
    '''
    data = open(filename, 'rU')
    fasta = OrderedDict()
    for line in data:
        if line[0] == '>':
            seqid = line[1:-1]
            assert seqid not in fasta # no repeats
            fasta[seqid] = ''
        else:
            l= line.split()
            if l:
                fasta[seqid] += l[0]
    data.close()
    return fasta



mhc_class_1_alfas = read_fasta(path_to_db / 'ClassI_prot.alfas')
# add HLA-G 2022-04-30
mhc_class_1_alfas.update(read_fasta(path_to_db / 'new_imgt_hla/G_prot.alfas'))
# add HLA-E 2022-05-03
mhc_class_1_alfas.update(read_fasta(path_to_db / 'new_imgt_hla/E_prot.alfas'))

ks = list(mhc_class_1_alfas.keys())
lencheck = None
for k in ks:
    newseq = mhc_class_1_alfas[k].replace('-', ALL_GENES_GAP_CHAR)\
             .replace('X',ALL_GENES_GAP_CHAR)
    mhc_class_1_alfas[k] = newseq
    if lencheck is None:
        lencheck = len(newseq)
    else:
        assert lencheck == len(newseq)

# read mouse sequences (no gaps in them)
# these mouse seqs are not the same length
mhc_class_1_alfas.update(read_fasta(path_to_db / 'mouse_class1_align.fasta'))

# v2 means new imgt hla class 2 human alignments/sequences:
mhc_class_2_alfas = {
    'A': read_fasta(path_to_db / 'both_class_2_A_chains_v2.alfas'),
    'B': read_fasta(path_to_db / 'both_class_2_B_chains_v2.alfas'),
}

assert all(all(all(x in amino_acids or x==ALL_GENES_GAP_CHAR
                   for x in seq)
               for seq in alfas.values())
           for alfas in [mhc_class_1_alfas, mhc_class_2_alfas['A'],
                         mhc_class_2_alfas['B']])

# the code below is now redundant since we moved the info files and had applied these
# changes already
#
# complex_pdb_dir = '/home/pbradley/csdat/tcrpepmhc/tcrdock_pdbs/'
# ternary_infofiles = [
#     complex_pdb_dir+'tcrdock_pdbs_info_class_1_2021-08-05_chainseqs_mhc_alseqs.tsv',
#     complex_pdb_dir+'tcrdock_pdbs_info_class_2_2021-08-05_trimmed_alignseqs.tsv',
# ]
# ternary_info = pd.concat([pd.read_table(x) for x in ternary_infofiles])
# ternary_info.set_index('pdbid', inplace=True, drop=False)
# def drop_extra_colons(allele):
#     if ',' in allele:
#         a,b = allele.split(',')
#         return drop_extra_colons(a)+','+drop_extra_colons(b)
#     else:
#         return ':'.join(allele.split(':')[:2])

# ternary_info['mhc_allele'] = ternary_info.mhc_allele.map(drop_extra_colons)

# # update mouse allele names-- want these to match the names in mhc_class_1_alfas
# #
# for big in 'DKL':
#     for little in 'BD':
#         old = f'{big}-{little}'
#         new = f'H2{big}{little.lower()}'
#         if old in ternary_info.mhc_allele.values:
#             assert new in mhc_class_1_alfas
#             ternary_info['mhc_allele'] = ternary_info.mhc_allele.replace(old, new)

# pmhc_infofile = (path_to_db / #'/home/pbradley/gitrepos/TCRpepMHC/tcrdock/db/'
#                  'pdb_mhc_info_class_1_2021-08-05_pdbfiles_alignseqs_w_EG.tsv')#NOTE EG
# pmhc_info = pd.read_table(pmhc_infofile).set_index('pdbid', drop=False)

# tcr_infofile = ('/home/pbradley/csdat/tcrpepmhc/amir/'
#                 'all_unbound_tcr_chain_info_both_v1.tsv') # mouse and human
# tcr_info = pd.read_table(tcr_infofile).set_index(['pdbid','ab'], drop=False)
# tcr_info['ternary'] = tcr_info.pdbid.isin(set(ternary_info.pdbid))

# TCR, PMHC, TERNARY = 'tcr', 'pmhc', 'ternary'
# all_template_info = {TCR:tcr_info, TERNARY:ternary_info, PMHC:pmhc_info}

TCR, PMHC, TERNARY = 'tcr', 'pmhc', 'ternary'

all_template_info = {}
for tag in [TCR, PMHC, TERNARY]:
    info = pd.read_table(path_to_db / f'{tag}_templates_v2.tsv')
    if tag == TCR:
        info.set_index(['pdbid','ab'], drop=False, inplace=True)
    else:
        info.set_index('pdbid', drop=False, inplace=True)
    all_template_info[tag] = info
tcr_info = all_template_info[TCR]
pmhc_info = all_template_info[PMHC]
ternary_info = all_template_info[TERNARY]

all_template_poses = {TCR:{}, PMHC:{}, TERNARY:{}}

BAD_DGEOM_PDBIDS = '5sws 7jwi 4jry 4nhu 3tjh 4y19 4y1a 1ymm 2wbj'.split()
BAD_PMHC_PDBIDS = '3rgv 4ms8 6v1a 6v19 6v18 6v15 6v13 6v0y 2uwe 2jcc 2j8u 1lp9'.split()
# 3rgv has cterm of peptide out of groove
# 4ms8 has chainbreaks (should check for those!)
# rest are all human MHC with mouse TCRs...
## grep ^both ~/tcr_scripts/tmp.pdb_tcr_mouse.2021-08-05.log  | awk '($4+$6+$8+$10<10)' | grep pdb_files | nsort 16 -r | awk '{print $2, $16}'
# 6v1a 82
# 6v19 82
# 6v18 82
# 6v15 82
# 6v13 82
# 6v0y 82
# 2uwe 72
# 2jcc 72
# 2j8u 72
# 1lp9 72


######################################################################################88
######################################################################################88
######################### END I/O ######################################################
######################################################################################88
######################################################################################88




_cached_tcrdisters = {}
def get_tcrdister(organism):
    if organism not in _cached_tcrdisters:
        _cached_tcrdisters[organism] = tcrdist.tcr_distances.TcrDistCalculator(organism)
    return _cached_tcrdisters[organism]


def blosum_align(
        seq1,
        seq2,
        gap_open=-11, # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3848038/
        gap_extend=-1,
        global_align=False,
        verbose=False,
):
    ''' return 0-indexed dictionary mapping from seq1 to seq2 positions
    '''

    scorematrix = matlist.blosum62

    if global_align:
        alignments = pairwise2.align.globalds(
            seq1, seq2, scorematrix, gap_open, gap_extend)
    else:
        alignments = pairwise2.align.localds(
            seq1, seq2, scorematrix, gap_open, gap_extend)
    alignment = max(alignments, key=lambda x:x.score)
    alseq1, alseq2, score, begin, end = alignment

    if verbose:
        print(format_alignment(*alignment))

    assert alseq1.replace('-','') == seq1
    assert alseq2.replace('-','') == seq2

    align = {}
    for i,(a,b) in enumerate(zip(alseq1, alseq2)):
        if a!= '-' and b!='-':
            #assert begin <= i <= end
            pos1 = i-alseq1[:i].count('-')
            pos2 = i-alseq2[:i].count('-')
            align[pos1] = pos2
    return align


# for building pmhc:TCR models from allele/v/j/cdr3 info

def get_v_seq_up_to_cys(organism, v_gene):
    vg = all_genes[organism][v_gene]

    alseq = vg.alseq
    cys_alpos = vg.cdr_columns[-1][0]-1 # they were 1-indexed, now 0-indexed
    if alseq[cys_alpos] != 'C':
        print('notC', alseq[cys_alpos])
    geneseq = alseq[:cys_alpos+1].replace(ALL_GENES_GAP_CHAR, '')
    return geneseq

def get_j_seq_after_cdr3(organism, j_gene):
    jg = all_genes[organism][j_gene]
    alseq = jg.alseq
    cdr3seq = jg.cdrs[0]
    assert alseq.startswith(cdr3seq)
    return alseq[len(cdr3seq):].replace(ALL_GENES_GAP_CHAR,'')

#A02_seq = 'GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQR'


## borrowed from cdr3s_human.py
## 1-indexed:
extra_alignment_columns_1x = { 'mouse':{'A':[9,86],'B':[] }, ## 1-indexed
                               'human':{'A':[],'B':[] } }


core_positions_generic_1x = [
    21, 23, 25,   ##  23 is C
    39, 41,       ##  41 is W
    53, 54, 55,
    78,           ##            maybe also 80?
    89,           ##  89 is L
    102, 103, 104 ## 104 is C
]

all_core_alseq_positions_0x = {}
for organism in extra_alignment_columns_1x:
    for ab,xcols in extra_alignment_columns_1x[organism].items():
        positions = [ x-1+sum(y<=x for y in xcols) # this is not quite right but it wrks
                      for x in core_positions_generic_1x ]

        if False:
            for v,g in all_genes[organism].items():
                if g.chain == ab and g.region == 'V' and v.endswith('*01'):
                    coreseq = ''.join([g.alseq[x] for x in positions])
                    print(coreseq, v, organism, ab)

        all_core_alseq_positions_0x[(organism,ab)] = positions


def get_core_positions_0x(organism, v_gene):
    vg = all_genes[organism][v_gene]

    alseq = vg.alseq

    core_positions_0x = []
    for pos in all_core_alseq_positions_0x[(organism, vg.chain)]:
        core_positions_0x.append(pos - alseq[:pos].count(ALL_GENES_GAP_CHAR))
    return core_positions_0x


def align_chainseq_to_imgt_msa(organism, chainseq, v_gene):
    #
    vg = all_genes[organism][v_gene]

    alseq = vg.alseq
    geneseq = get_v_seq_up_to_cys(organism, v_gene)

    # extends past end of geneseq
    geneseq_to_alseq = {k-alseq[:k].count(ALL_GENES_GAP_CHAR) : k
                        for k,a in enumerate(alseq) if a!=ALL_GENES_GAP_CHAR}

    chainseq_to_geneseq = blosum_align(chainseq, geneseq)

    chainseq_to_alseq = {i:geneseq_to_alseq[j]
                         for i,j in chainseq_to_geneseq.items()}

    return chainseq_to_alseq


def align_chainseq_to_structure_msa(organism, chainseq, v_gene, msa_type='both'):
    assert msa_type in ['both','human']

    if msa_type == 'human':
        assert organism == 'human'
        #assert v_gene in human_structure_alignments.index
        row = human_structure_alignments.loc[v_gene]
    else:
        row = both_structure_alignments.loc[(organism,v_gene)]

    alseq = row.alseq

    geneseq = get_v_seq_up_to_cys(organism, v_gene)
    assert alseq.replace(ALL_GENES_GAP_CHAR,'') == geneseq

    #
    geneseq_to_alseq = {k-alseq[:k].count(ALL_GENES_GAP_CHAR) : k
                        for k,a in enumerate(alseq) if a!=ALL_GENES_GAP_CHAR}

    chainseq_to_geneseq = blosum_align(chainseq, geneseq)

    chainseq_to_alseq = {i:geneseq_to_alseq[j]
                         for i,j in chainseq_to_geneseq.items()}

    return chainseq_to_alseq

_tcr_alignment_cache = None
def align_tcr_info_pdb_chain_to_structure_msa(pdbid, ab, msa_type_in):
    ''' pdbid,ab has to be in the tcr_info index
    '''
    global _tcr_alignment_cache
    assert msa_type_in in ['both','human']
    if _tcr_alignment_cache is None:
        _tcr_alignment_cache = {
            'both' :{'A':{}, 'B':{}},
            'human':{'A':{}, 'B':{}}
        }
        print('setting up cache for align_tcr_info_pdb_chain_to_structure_msa function')
        for l in tcr_info.itertuples():
            geneseq = get_v_seq_up_to_cys(l.organism, l.v_gene)
            for msa_type in ['both', 'human']:
                if msa_type == 'human':
                    if l.organism != 'human':
                        continue
                    row = human_structure_alignments.loc[l.v_gene]
                else:
                    row = both_structure_alignments.loc[(l.organism, l.v_gene)]
                alseq = row.alseq
                assert alseq.replace(ALL_GENES_GAP_CHAR,'') == geneseq
                geneseq_to_alseq = {k-alseq[:k].count(ALL_GENES_GAP_CHAR) : k
                                    for k,a in enumerate(alseq)
                                    if a!=ALL_GENES_GAP_CHAR}

                chainseq_to_geneseq = blosum_align(l.chainseq, geneseq)

                chainseq_to_alseq = {i:geneseq_to_alseq[j]
                                     for i,j in chainseq_to_geneseq.items()}
                _tcr_alignment_cache[msa_type][l.ab][l.pdbid] = chainseq_to_alseq

        print('DONE setting up cache for align_tcr_info_pdb_chain_to_structure_msa',
              'function')
    return _tcr_alignment_cache[msa_type_in][ab][pdbid]



def align_cdr3s(a, b):
    shortseq, longseq = (a,b) if len(a)<len(b) else (b,a)
    lenshort, lenlong = len(shortseq), len(longseq)
    assert lenshort >= 5
    gappos = min( 6, 3 + (lenshort-5)//2 )
    num_gaps = lenlong-lenshort
    align = {i:i for i in range(gappos)}
    align.update({i:i+num_gaps for i in range(gappos,lenshort)})
    if len(a)>=len(b): # reverse
        align = {j:i for i,j in align.items()}
    return align



def get_tcr_chain_trim_positions(organism, chainseq, v, j, cdr3):
    ''' could include None if the position is missing (deleted) for this v gene...
    '''
    assert cdr3 in chainseq
    nterm_segment_alseq_positions_0x = [3, 4, 5]
    # 1. imgt alignment 1-index positions 4-6 ([VIL].Q)
    # 2. 1x core positions 0-->9
    # 3. residue before core pos 10 --> residue after GXG (CDR3 + 4 rsds after)
    vg = all_genes[organism][v]

    alseq = vg.alseq

    geneseq = get_v_seq_up_to_cys(organism, v)
    assert alseq.replace(ALL_GENES_GAP_CHAR,'').startswith(geneseq)

    geneseq_to_chainseq = blosum_align(geneseq, chainseq)

    core_positions_0x = get_core_positions_0x(organism, v)
    assert len(core_positions_0x) == TCR_CORE_LEN

    geneseq_positions = []
    for pos in nterm_segment_alseq_positions_0x:
        if alseq[pos] == ALL_GENES_GAP_CHAR:
            print('N None:', organism, v)
            geneseq_positions.append(None)
        else:
            geneseq_positions.append(pos-alseq[:pos].count(ALL_GENES_GAP_CHAR))

    start, stop = core_positions_0x[0], core_positions_0x[9] # inclusive!
    geneseq_positions.extend(range(start, stop+1))

    chainseq_positions = [geneseq_to_chainseq.get(pos,None)
                          for pos in geneseq_positions]

    start = geneseq_to_chainseq[core_positions_0x[10]-1]
    stop = chainseq.index(cdr3)+len(cdr3)+3 # index of position after GXG
    chainseq_positions.extend(range(start, stop+1))

    if None in chainseq_positions:
        print(organism, v, [i for i,x in enumerate(chainseq_positions) if x is None])

        blosum_align(geneseq, chainseq, verbose=True)


    return chainseq_positions




def align_vgene_to_template_pdb_chain(
        ab,
        trg_organism,
        trg_v,
        trg_j,
        trg_cdr3,
        trg_msa_alignments,
        tmp_pdbid,
        # tmp_organism,
        # tmp_v,
        # tmp_j,
        # tmp_cdr3,
        # tmp_chainseq,
        verbose=False,
):
    ''' Returns trg_to_tmp, trg_chainseq

    trg_to_tmp is 0-indexed dict aligning trg_chainseq to tmp_chainseq

    '''
    # align trg_vg to nearest rep (trg_rep) in structure MSA
    # align tmp_vg to nearest rep (tmp_rep) in structure MSA
    # use structure MSA to create alignment between trg_rep and tmp_rep
    # use IMGT MSA to align before the first cys??
    # align cdr3 with gaps in the middle
    # align jgenes

    tmp_row = tcr_info.loc[(tmp_pdbid,ab)]
    tmp_organism = tmp_row.organism
    tmp_chainseq = tmp_row.chainseq
    tmp_cdr3 = tmp_row.cdr3

    trg_vseq = get_v_seq_up_to_cys(trg_organism, trg_v)
    trg_jseq = get_j_seq_after_cdr3(trg_organism, trg_j)

    trg_chainseq = trg_vseq[:-1] + trg_cdr3 + trg_jseq

    # align V regions
    msa_type = ('human' if trg_organism=='human' and tmp_organism=='human' else
                'both')

    trg_vseq_to_alseq = trg_msa_alignments[msa_type]

    # trg_vseq_to_alseq_old = align_chainseq_to_structure_msa(
    #     trg_organism, trg_chainseq, trg_v, msa_type=msa_type)

    # assert (sorted(trg_vseq_to_alseq.items()) ==
    #         sorted(trg_vseq_to_alseq_old.items()))
    #print(tmp_pdbid, tmp_organism, ab, msa_type)
    tmp_vseq_to_alseq = align_tcr_info_pdb_chain_to_structure_msa(
        tmp_pdbid, ab, msa_type)

    # tmp_vseq_to_alseq_old = align_chainseq_to_structure_msa(
    #     tmp_organism, tmp_chainseq, tmp_row.v_gene, msa_type=msa_type)
    # for i in sorted(set(tmp_vseq_to_alseq.keys())|
    #                 set(tmp_vseq_to_alseq_old.keys())):
    #     j = tmp_vseq_to_alseq.get(i,-1)
    #     k = tmp_vseq_to_alseq_old.get(i,-1)
    #     print(f'{tmp_row.chainseq[i]} {i:4d} {j:4d} {k:4d}')
    # assert (sorted(tmp_vseq_to_alseq.items()) ==
    #         sorted(tmp_vseq_to_alseq_old.items()))

    alseq_to_tmp_vseq = {j:i for i,j in tmp_vseq_to_alseq.items()}

    trg_to_tmp = {} # alignment over this TCR chain, 0-indexed
    for i,j in trg_vseq_to_alseq.items():
        if j in alseq_to_tmp_vseq:
            trg_to_tmp[i]=alseq_to_tmp_vseq[j]

    # align cdr3 regions
    assert trg_vseq[-1] == 'C'
    trg_offset = len(trg_vseq)-1 # skip 'C'
    tmp_offset = tmp_chainseq.index(tmp_cdr3)

    al = align_cdr3s(trg_cdr3, tmp_cdr3)
    trg_to_tmp.update({i+trg_offset:j+tmp_offset
                       for i,j in al.items()})
    trg_offset += len(trg_cdr3)
    tmp_offset += len(tmp_cdr3)


    # align j region
    tmp_jseq= tmp_chainseq[tmp_chainseq.index(tmp_cdr3)+
                           len(tmp_cdr3):]
    if (trg_jseq[:3]+tmp_jseq[:3]).count('G') < 4:
        print('GXGs?', trg_jseq[:3], tmp_jseq[:3])
    for i in range(min(len(trg_jseq), len(tmp_jseq))):
        trg_to_tmp[trg_offset+i] = tmp_offset+i

    identities = sum(trg_chainseq[i]==tmp_chainseq[j]
                     for i,j in trg_to_tmp.items())/len(trg_chainseq)
    if verbose:
        print(f'align_vgene_to_template_pdb_chain: {identities:6.3f}',
              trg_v, tmp_v, trg_cdr3, tmp_cdr3, verbose=verbose)

    return trg_to_tmp, trg_chainseq


# def old_align_vgene_to_template_pdb_chain(
#         ab,
#         trg_organism,
#         trg_v,
#         trg_j,
#         trg_cdr3,
#         tmp_organism,
#         tmp_v,
#         tmp_j,
#         tmp_cdr3,
#         tmp_chainseq,
#         verbose=False,
# ):
#     ''' Returns trg_to_tmp, trg_chainseq

#     trg_to_tmp is 0-indexed dict aligning trg_chainseq to tmp_chainseq

#     '''
#     # align trg_vg to nearest rep (trg_rep) in structure MSA
#     # align tmp_vg to nearest rep (tmp_rep) in structure MSA
#     # use structure MSA to create alignment between trg_rep and tmp_rep
#     # use IMGT MSA to align before the first cys??
#     # align cdr3 with gaps in the middle
#     # align jgenes

#     trg_vseq = get_v_seq_up_to_cys(trg_organism, trg_v)
#     trg_jseq = get_j_seq_after_cdr3(trg_organism, trg_j)

#     trg_chainseq = trg_vseq[:-1] + trg_cdr3 + trg_jseq

#     # align V regions
#     msa_type = ('human' if trg_organism=='human' and tmp_organism=='human' else
#                 'both')
#     trg_vseq_to_alseq = align_chainseq_to_structure_msa(
#         trg_organism, trg_chainseq, trg_v, msa_type=msa_type)

#     tmp_vseq_to_alseq = align_chainseq_to_structure_msa(
#         tmp_organism, tmp_chainseq, tmp_v, msa_type=msa_type)

#     alseq_to_tmp_vseq = {j:i for i,j in tmp_vseq_to_alseq.items()}

#     trg_to_tmp = {} # alignment over this TCR chain, 0-indexed
#     for i,j in trg_vseq_to_alseq.items():
#         if j in alseq_to_tmp_vseq:
#             trg_to_tmp[i]=alseq_to_tmp_vseq[j]

#     # align cdr3 regions
#     assert trg_vseq[-1] == 'C'
#     trg_offset = len(trg_vseq)-1 # skip 'C'
#     tmp_offset = tmp_chainseq.index(tmp_cdr3)

#     al = align_cdr3s(trg_cdr3, tmp_cdr3)
#     trg_to_tmp.update({i+trg_offset:j+tmp_offset
#                        for i,j in al.items()})
#     trg_offset += len(trg_cdr3)
#     tmp_offset += len(tmp_cdr3)


#     # align j region
#     tmp_jseq= tmp_chainseq[tmp_chainseq.index(tmp_cdr3)+
#                            len(tmp_cdr3):]
#     if (trg_jseq[:3]+tmp_jseq[:3]).count('G') < 4:
#         print('GXGs?', trg_jseq[:3], tmp_jseq[:3])
#     for i in range(min(len(trg_jseq), len(tmp_jseq))):
#         trg_to_tmp[trg_offset+i] = tmp_offset+i

#     identities = sum(trg_chainseq[i]==tmp_chainseq[j]
#                      for i,j in trg_to_tmp.items())/len(trg_chainseq)
#     if verbose:
#         print(f'align_vgene_to_template_pdb_chain: {identities:6.3f}',
#               trg_v, tmp_v, trg_cdr3, tmp_cdr3, verbose=verbose)

#     return trg_to_tmp, trg_chainseq


def get_mhc_class_1_alseq(allele):
    if allele in mhc_class_1_alfas:
        return mhc_class_1_alfas[allele]
    sortl = []
    for k in mhc_class_1_alfas:
        if k.startswith(allele) and k[len(allele)] == ':':
            suffix = [int(x) if x.isdigit() else 100
                      for x in k[len(allele)+1:].split(':')]
            sortl.append((suffix, k))
    if sortl:
        sortl.sort()
        best_allele = sortl[0][1]
        #print('close allele:', allele, best_allele)
        return mhc_class_1_alfas[best_allele]
    elif allele.count(':')>1:
        trim_allele = ':'.join(allele.split(':')[:2])
        return get_mhc_class_1_alseq(trim_allele)
    else:
        return None

def get_mhc_class_2_alseq(chain, allele):
    if allele in mhc_class_2_alfas[chain]:
        return mhc_class_2_alfas[chain][allele]
    assert '*' in allele # human
    sortl = []
    for k in mhc_class_2_alfas[chain]:
        if k.startswith(allele) and k[len(allele)] == ':':
            suffix = [int(x) if x.isdigit() else 100
                      for x in k[len(allele)+1:].split(':')]
            sortl.append((suffix, k))
    if sortl:
        sortl.sort()
        best_allele = sortl[0][1]
        #print('close allele:', allele, best_allele)
        return mhc_class_2_alfas[chain][best_allele]
    elif allele.count(':')>1:
        trim_allele = ':'.join(allele.split(':')[:2])
        return get_mhc_class_2_alseq(chain, trim_allele)
    else:
        return None


def get_template_pose_and_tdinfo(pdbid, complex_type):
    ''' returns pose, tdinfo
    complex_type should be in {TCR, TERNARY, PMHC}
    '''
    info, poses = all_template_info[complex_type], all_template_poses[complex_type]
    if pdbid not in poses:
        pdbfile = set(info[info.pdbid==pdbid].pdbfile)
        if not pdbfile:
            # right now we only have class 1 in the special pmhc info...
            assert complex_type == PMHC
            info = all_template_info[TERNARY] # NOTE NOTE NOTE
            pdbfile = set(info[info.pdbid==pdbid].pdbfile)
        assert len(pdbfile) == 1
        pdbfile = pdbfile.pop()
        #print('read:', pdbfile)
        pose = pdblite.pose_from_pdb(pdbfile)
        tdifile = pdbfile+'.tcrdock_info.json'
        tdinfo = TCRdockInfo().from_string(open(tdifile,'r').read())
        #print('make tdinfo 0-indexed:', tdifile)
        #tdinfo.renumber({i+1:i for i in range(len(pose['sequence']))})
        poses[pdbid] = (pose, tdinfo)
    pose, tdinfo = poses[pdbid]
    return copy.deepcopy(pose), TCRdockInfo().from_dict(tdinfo.to_dict())# return copies

def count_peptide_mismatches(a,b):
    if len(a)>len(b):
        return count_peptide_mismatches(b,a)
    lendiff = len(b)-len(a)
    assert lendiff>=0
    min_mismatches = len(a)
    for shift in range(lendiff+1):
        mismatches = sum(x!=y for x,y in zip(a,b[shift:]))
        if mismatches < min_mismatches:
            min_mismatches = mismatches
    # also allow bulging out in the middle
    nt = len(a)//2
    ct = len(a) - nt
    btrim = b[:nt]+b[-ct:]
    assert len(btrim) == len(a)
    mismatches = sum(x!=y for x,y in zip(a,btrim))
    if mismatches < min_mismatches:
        min_mismatches = mismatches
    return min_mismatches

pep1 = 'DSIODJSJD' # sanity checking...
pep2 = 'DSIXDJSJD'
assert count_peptide_mismatches(pep1,pep1)==0
assert count_peptide_mismatches(pep1,pep1[:-1])==0
assert count_peptide_mismatches(pep1,pep2)==1

def get_clean_and_nonredundant_ternary_tcrs_df(
        min_peptide_mismatches = 3,
        min_tcrdist = 120.5,
        peptide_tcrdist_logical = 'or', # 'or' or 'and'
        drop_HLA_E = True,
        verbose=False,
        skip_redundancy_check=False, # just add resol,mismatches and sort
        filter_BAD_DGEOM_PDBIDS=True,
        filter_BAD_PMHC_PDBIDS=True,
):
    ''' peptide_tcrdist_logical = 'or' is more stringent redundancy filtering
    ie, smaller df returned. A TCR:pMHC is considered redundant by peptide OR
    by TCRdist.
    'and' means redundant only if both peptide and TCRdist redundant
    '''
    assert peptide_tcrdist_logical in ['or','and']

    tcrs = ternary_info.copy()

    bad_pdbids = set()
    if filter_BAD_DGEOM_PDBIDS:
        bad_pdbids.update(set(BAD_DGEOM_PDBIDS))
    if filter_BAD_PMHC_PDBIDS:
        bad_pdbids.update(set(BAD_PMHC_PDBIDS))
    print('get_clean_and_nonredundant_ternary_tcrs_df: bad_pdbids', bad_pdbids)

    tcrs = tcrs[~tcrs.pdbid.isin(bad_pdbids)]

    if drop_HLA_E: # drop HLA-E
        tcrs = tcrs[~tcrs.mhc_allele.str.startswith('E*')]

    # need to add resolution
    pdbid2resolution = {}
    pdbid2mismatches = {}
    for organism in 'human mouse'.split():
        # temporary hack...
        logfile = path_to_db / f'tmp.pdb_tcr_{organism}.2021-08-05.log'
        assert exists(logfile)
        for line in os.popen(f'grep ^both {logfile}'):
            l = line.split()
            pdbid = l[1]
            resolution = float(l[-2])
            v_mismatches = int(l[3]) + int(l[7])
            pdbid2resolution[pdbid] = resolution
            pdbid2mismatches[pdbid] = min(v_mismatches, pdbid2mismatches.get(pdbid,100))

    tcrs['resolution'] = tcrs.pdbid.map(pdbid2resolution)
    tcrs['mismatches'] = tcrs.pdbid.map(pdbid2mismatches)
    assert tcrs.resolution.isna().sum()==0

    ## remove redundancy
    # sort by count for peptide, then resolution

    tcrs['org_pep'] = (tcrs.mhc_class.astype(str) + "_" +
                       tcrs.organism + "_" + tcrs.pep_seq)
    tcrs['org_pep_count'] = tcrs.org_pep.map(tcrs.org_pep.value_counts())

    tcrs['neg_quality'] = -10*tcrs.org_pep_count + tcrs.resolution + 0.5*tcrs.mismatches

    tcrs.sort_values('neg_quality', inplace=True)

    if not skip_redundancy_check:
        tdist = {'human': tcrdist.tcr_distances.TcrDistCalculator('human'),
                 'mouse': tcrdist.tcr_distances.TcrDistCalculator('mouse')}

        tcr_tuples = [((l.va, l.ja, l.cdr3a), (l.vb, l.jb, l.cdr3b))
                      for l in tcrs.itertuples()]

        is_redundant = [False]*tcrs.shape[0]

        for i, irow in enumerate(tcrs.itertuples()):
            # check to see if i is too close to any previous tcr
            for j in range(i):
                jrow = tcrs.iloc[j]
                if is_redundant[j]:
                    continue

                pep_mms = count_peptide_mismatches(irow.pep_seq, jrow.pep_seq)
                if (irow.organism == jrow.organism and
                    irow.mhc_class == jrow.mhc_class):
                    tcrd = tdist[irow.organism](tcr_tuples[i], tcr_tuples[j])
                    pep_red = (pep_mms < min_peptide_mismatches)
                    tcr_red = (tcrd < min_tcrdist)
                    if ((peptide_tcrdist_logical=='or' and
                         (pep_red or tcr_red)) or
                        (peptide_tcrdist_logical=='and' and
                         (pep_red and tcr_red))):

                        # redundant
                        is_redundant[i] = True
                        if verbose:
                            print('skip:', i, irow.org_pep, irow.neg_quality,
                                  tcr_tuples[i])
                        break

        tcrs['is_redundant'] = is_redundant

        tcrs = tcrs[~tcrs.is_redundant].copy()
    return tcrs

def filter_templates_by_tcrdist(
        templates, # dataframe
        organism, va, cdr3a, vb, cdr3b,
        min_paired_tcrdist=-1,
        min_singlechain_tcrdist=-1,
):
    ''' returns new templates

    will have paired_tcrdist and singlechain_tcrdist cols
    '''
    tcrdister = get_tcrdister('human_and_mouse')

    template_tcrs = [
        ((l.organism[0]+l.va, None, l.cdr3a), (l.organism[0]+l.vb, None, l.cdr3b))
        for l in templates.itertuples()]

    target_tcr = ((organism[0]+va, None, cdr3a), (organism[0]+vb, None, cdr3b))

    templates['paired_tcrdist'] = np.array(
        [tcrdister(target_tcr,x) for x in template_tcrs])

    for iab, ab in enumerate('AB'):
        templates[ab+'_tcrdist'] = np.array(
            [tcrdister.single_chain_distance(target_tcr[iab], x[iab])
             for x in template_tcrs])

    templates['AB_tcrdist'] = templates.paired_tcrdist

    templates['singlechain_tcrdist'] = np.minimum(
        templates.A_tcrdist, templates.B_tcrdist)

    too_close_mask = ((templates.organism==organism)&
                      ((templates.paired_tcrdist < min_paired_tcrdist)|
                       (templates.singlechain_tcrdist < min_singlechain_tcrdist)))
    print('too close by tcrdist:', np.sum(too_close_mask), organism,
          va, cdr3a, vb, cdr3b)

    return templates[~too_close_mask].copy()

def filter_templates_by_peptide_mismatches(
        templates,
        organism,
        mhc_class,
        peptides_for_filtering,
        min_peptide_mismatches,
):
    if not peptides_for_filtering:
        return templates.copy()

    templates['filt_peptide_mismatches'] = np.array(
        [min(count_peptide_mismatches(p, l.pep_seq) for p in peptides_for_filtering)
             for l in templates.itertuples()])
    too_close_mask = ((templates.organism==organism) &
                      (templates.mhc_class==mhc_class) &
                      (templates.filt_peptide_mismatches<min_peptide_mismatches))
    print('too close by peptide mismatches:', np.sum(too_close_mask), organism,
          peptides_for_filtering)

    return templates[~too_close_mask].copy()

def pick_dgeom_templates(
        num_templates, # per chain setting
        organism,
        mhc_class,
        mhc_allele,
        peptide,
        va,
        cdr3a,
        vb,
        cdr3b,
        min_peptide_mismatches=3, # ie, 3 or more is OK
        min_paired_tcrdist=96.5, # 96 or less is BAD
        min_singlechain_tcrdist=35.9, # ie 36 or more is OK
        min_template_template_paired_tcrdist = 35.9, # for redundancy filtering
        peptides_for_filtering = None,
        drop_HLA_E = True,
        exclude_pdbids = None,
):
    ''' returns dict
    {
      'A' : alpha_templates_df,
      'B' : beta_templates_df,
      'AB': paired_templates_df,
    }

    '''
    if peptides_for_filtering is None:
        peptides_for_filtering = []
    elif peptides_for_filtering:
        assert peptide in peptides_for_filtering # sanity? seems like should be true

    if exclude_pdbids is None:
        exclude_pdbids = []
    else:
        exclude_pdbids = frozenset(exclude_pdbids)

    print('pick_dgeom_templates:', min_peptide_mismatches, min_paired_tcrdist,
          min_singlechain_tcrdist, min_template_template_paired_tcrdist,
          peptides_for_filtering)

    tcrdister = get_tcrdister('human_and_mouse')

    # sorts by pep-count, resolution, mismatches
    # excludes BAD_DGEOM_PDBIDS but not BAD_PMHC_PDBIDS
    templates = get_clean_and_nonredundant_ternary_tcrs_df(
        skip_redundancy_check=True, filter_BAD_PMHC_PDBIDS=False,
        drop_HLA_E=drop_HLA_E)

    templates = templates[templates.mhc_class == mhc_class]
    templates = templates[~templates.pdbid.isin(exclude_pdbids)]

    templates = filter_templates_by_peptide_mismatches(
        templates, organism, mhc_class, peptides_for_filtering,
        min_peptide_mismatches)

    templates = filter_templates_by_tcrdist(
        templates, organism, va, cdr3a, vb, cdr3b,
        min_paired_tcrdist=min_paired_tcrdist,
        min_singlechain_tcrdist=min_singlechain_tcrdist,
    )

    # currently unused...
    templates['peptide_mismatches'] = np.array(
        [count_peptide_mismatches(peptide, l.pep_seq)
         for l in templates.itertuples()])

    def calc_chain_tcrdist(row1, row2, chain, tcrdister=tcrdister):
        tcr1 = ((row1.organism[0]+row1.va, None, row1.cdr3a),
                (row1.organism[0]+row1.vb, None, row1.cdr3b))
        tcr2 = ((row2.organism[0]+row2.va, None, row2.cdr3a),
                (row2.organism[0]+row2.vb, None, row2.cdr3b))
        if chain == 'AB':
            return tcrdister(tcr1, tcr2)
        else:
            assert chain in ['A','B']
            ich = 'AB'.index(chain)
            return tcrdister.single_chain_distance(tcr1[ich], tcr2[ich])


    all_templates = {}
    for chain in ['A','B','AB']:
        # sort by
        templates.sort_values(chain+'_tcrdist', inplace=True)

        dfl = []
        for _, row1 in templates.iterrows():
            # check for redundancy
            redundant = False
            for row2 in dfl:
                if (calc_chain_tcrdist(row1, row2, 'AB') <
                    min_template_template_paired_tcrdist):
                    redundant = True
                    print('redundant template:', chain, row1.pdbid, row2.pdbid)
                    break
            if redundant:
                continue
            dfl.append(row1.copy())
            print('new_dgeom_template:', chain, len(dfl), row1[chain+'_tcrdist'],
                  row1.organism, row1.mhc_class, row1.mhc_allele, row1.pep_seq,
                  row1.va, row1.cdr3a, row1.vb, row1.cdr3b)
            if len(dfl)>=num_templates:
                break
        all_templates[chain] = pd.DataFrame(dfl)

    return all_templates






def make_templates_for_alphafold_same_pmhc(
        organism,
        va,
        ja,
        cdr3a,
        vb,
        jb,
        cdr3b,
        mhc_class,
        mhc_allele,
        peptide,
        outfile_prefix,
        num_runs=3,
        num_templates_per_run=4,
        min_single_chain_tcrdist=-1,
        next_best_identity_threshold_mhc=0.98,
        next_best_identity_threshold_tcr=0.98,
        verbose=False,
):
    ''' Only use ternary templates: no franken templates

    Exclude potential templates based on

    - overall seqid

    OR

    - paired tcrdist

    right now, just a single run

    Make num_runs alignfiles <outfile_prefix>_<run>_alignments.tsv

    returns df with lines for "targets.tsv" file, last 2 columns are
    <alignfile> and <target_chainseq>

    returns None for failure

    '''
    # check arguments
    if mhc_class == 2:
        assert len(peptide) == CLASS2_PEPLEN
        assert mhc_allele.count(',') == 1

    tcrdister = get_tcrdister('human_and_mouse')

    core_len = TCR_CORE_LEN

    template_mask = ((ternary_info.organism == organism) &
                     (ternary_info.mhc_class == mhc_class) &
                     (ternary_info.mhc_allele == mhc_allele) &
                     (ternary_info.pep_seq == peptide) &
                     (~ternary_info.pdbid.isin(BAD_PMHC_PDBIDS))&
                     (~ternary_info.pdbid.isin(BAD_DGEOM_PDBIDS)))
    if not template_mask.sum():
        print('ERROR no matching templates:: make_templates_for_alphafold_same_pmhc',
              organism, mhc_class, mhc_allele, peptide)
    templates = ternary_info[template_mask]
    print('make_templates_for_alphafold_same_pmhc num_templates=', templates.shape[0])


    # get MHC align-seqs
    if mhc_class == 1:
        trg_mhc_alseqs = [get_mhc_class_1_alseq(mhc_allele)]
    else:
        trg_mhc_alseqs = [get_mhc_class_2_alseq('A', mhc_allele.split(',')[0]),
                          get_mhc_class_2_alseq('B', mhc_allele.split(',')[1])]


    return None

def align_vgene_to_structure_msas(organism, v_gene):
    msa_alignments = {}

    vseq = get_v_seq_up_to_cys(organism, v_gene)
    for msa_type in ['both','human']:
        if msa_type == 'human':
            if organism != 'human':
                continue
            row = human_structure_alignments.loc[v_gene]
        else:
            row = both_structure_alignments.loc[(organism, v_gene)]
        assert row.alseq.replace(ALL_GENES_GAP_CHAR,'') == vseq
        vseq_to_alseq = {k-row.alseq[:k].count(ALL_GENES_GAP_CHAR) : k
                         for k,a in enumerate(row.alseq) if a!=ALL_GENES_GAP_CHAR}
        msa_alignments[msa_type] = vseq_to_alseq
    return msa_alignments


def make_templates_for_alphafold(
        organism,
        va,
        ja,
        cdr3a,
        vb,
        jb,
        cdr3b,
        mhc_class,
        mhc_allele,
        peptide,
        outfile_prefix,
        num_runs=5, # match default in setup_for_alphafold
        num_templates_per_run=4,
        exclude_self_peptide_docking_geometries=False,
        exclude_docking_geometry_peptides=None, # None or list of peptides
        # below is only applied if exclude_docking_geometry_peptides is nonempty
        #   or exclude_self_peptide_docking_geometries
        min_dgeom_peptide_mismatches=3, # not used if exclude_* are False/None
        min_dgeom_paired_tcrdist=-1,
        min_dgeom_singlechain_tcrdist=-1,
        min_single_chain_tcrdist=-1,
        min_pmhc_peptide_mismatches=-1,
        next_best_identity_threshold_mhc=0.98,
        next_best_identity_threshold_tcr=0.98,
        ternary_bonus=0.05, # for tcr templates, in frac identities
        alt_self_peptides=None, # or list of peptides
        verbose=False,
        pick_dgeoms_using_tcrdist=False, # implies num_runs=3
        use_same_pmhc_dgeoms=False,
        exclude_pdbids=None,
):
    ''' Makes num_templates_per_run * num_runs template pdb files

    Make num_runs alignfiles <outfile_prefix>_<run>_alignments.tsv

    returns df with lines for "targets.tsv" file, last 2 columns are
    <alignfile> and <target_chainseq>

    returns None for failure

    '''
    from .pdblite import (apply_transform_Rx_plus_v, delete_chains, append_chains,
                          dump_pdb)

    if exclude_pdbids is None:
        exclude_pdbids = []
    else:
        exclude_pdbids = frozenset(exclude_pdbids)
        print(f'will exclude {len(exclude_pdbids)} pdbids')

    # check arguments
    if mhc_class == 2:
        assert len(peptide) == CLASS2_PEPLEN
        assert mhc_allele.count(',') == 1

    if pick_dgeoms_using_tcrdist:
        assert num_runs == 3 # AB, A, B

    if exclude_docking_geometry_peptides is None:
        exclude_docking_geometry_peptides = []
    if alt_self_peptides is None:
        alt_self_peptides = []

    if exclude_self_peptide_docking_geometries:
        # dont modify the passed-in list
        exclude_docking_geometry_peptides = (
            exclude_docking_geometry_peptides+[peptide]+alt_self_peptides)

    tcrdister = get_tcrdister('human_and_mouse')


    def show_alignment(al,seq1,seq2):
        if verbose:
            for i,j in sorted(al.items()):
                a,b = seq1[i], seq2[j]
                star = '*' if a==b else ' '
                print(f'{i:4d} {j:4d} {a} {star} {b}')
            idents = sum(seq1[i] == seq2[j] for i,j in al.items())/len(seq1)
            print(f'idents: {idents:6.3f}')

    core_len = TCR_CORE_LEN

    if mhc_class == 1:
        if organism=='human':
            # now adding HLA-E 2022-05-03
            assert mhc_allele[0] in 'ABCE' and mhc_allele[1]=='*' and ':' in mhc_allele
            mhc_allele = ':'.join(mhc_allele.split(':')[:2]) # just the 4 digits
        else:
            assert mhc_allele.startswith('H2') and mhc_allele in mhc_class_1_alfas

        # first: MHC part
        trg_mhc_alseq = get_mhc_class_1_alseq(mhc_allele)
        trg_mhc_seq = trg_mhc_alseq.replace(ALL_GENES_GAP_CHAR,'')

        sortl = []

        # use new pmhc-only data
        for l in pmhc_info.itertuples():
            if (l.organism!=organism or l.mhc_class!=mhc_class or
                l.pdbid in BAD_PMHC_PDBIDS or l.pdbid in exclude_pdbids):
                continue

            tmp_mhc_alseq = l.mhc_alignseq
            assert len(trg_mhc_alseq) == len(tmp_mhc_alseq)

            mhc_idents = sum(a==b and a!=ALL_GENES_GAP_CHAR
                             for a,b in zip(trg_mhc_alseq, tmp_mhc_alseq))

            if len(peptide) == len(l.pep_seq):
                pep_idents = sum(a==b for a,b in zip(peptide,l.pep_seq))
            else:
                pep_idents = sum(a==b for a,b in zip(peptide[:3]+peptide[-3:],
                                                     l.pep_seq[:3]+l.pep_seq[-3:]))
            mismatches_for_excluding = min(
                count_peptide_mismatches(x, l.pep_seq)
                for x in [peptide]+alt_self_peptides)
            if mismatches_for_excluding < min_pmhc_peptide_mismatches:
                if verbose:
                    print('peptide too close:', peptide, l.pep_seq,
                          'mismatches_for_excluding:', mismatches_for_excluding,
                          alt_self_peptides)
                continue
            assert len(peptide)-pep_idents >= min_pmhc_peptide_mismatches #sanity
            total = len(peptide)+len(trg_mhc_seq)
            frac = (mhc_idents+pep_idents)/total - 0.01*l.mhc_total_chainbreak
            sortl.append((frac, l.Index))

        sortl.sort(reverse=True)
        max_idents = sortl[0][0]
        print(f'mhc max_idents: {max_idents:.3f}', mhc_allele, peptide,
              pmhc_info.loc[sortl[0][1], 'mhc_allele'],
              pmhc_info.loc[sortl[0][1], 'pep_seq'])

        pmhc_alignments = []
        for (idents, pdbid) in sortl[:num_templates_per_run]:
            if idents < next_best_identity_threshold_tcr*max_idents:
                break
            templatel = pmhc_info.loc[pdbid]
            tmp_mhc_alseq = templatel.mhc_alignseq
            tmp_seql = templatel.chainseq.split('/')
            assert len(tmp_seql)==2 and tmp_seql[1] == templatel.pep_seq
            tmp_mhc_seq = tmp_seql[0] # class 1
            tmp_mhc_alseq_seq = tmp_mhc_alseq.replace(ALL_GENES_GAP_CHAR,'')
            assert tmp_mhc_alseq_seq in tmp_mhc_seq
            npad = tmp_mhc_seq.index(tmp_mhc_alseq_seq)
            #al1 = blosum_align(tmp_mhc_seq, tmp_mhc_alseq_seq)
            al1 = {i+npad:i for i in range(len(tmp_mhc_alseq_seq))}

            al2 = {i-tmp_mhc_alseq[:i].count(ALL_GENES_GAP_CHAR):
                   i-trg_mhc_alseq[:i].count(ALL_GENES_GAP_CHAR)
                   for i,(a,b) in enumerate(zip(tmp_mhc_alseq, trg_mhc_alseq))
                   if a != ALL_GENES_GAP_CHAR and b != ALL_GENES_GAP_CHAR}

            tmp_to_trg = {x:al2[y] for x,y in al1.items() if y in al2}
            trg_to_tmp = {y:x for x,y in tmp_to_trg.items()}

            trg_offset = len(trg_mhc_seq)
            tmp_offset = len(tmp_mhc_seq)
            trg_peplen, tmp_peplen = len(peptide), len(templatel.pep_seq)
            if trg_peplen == tmp_peplen:
                for i in range(trg_peplen):
                    trg_to_tmp[trg_offset+i] = tmp_offset+i
            else:
                for i in range(3):
                    trg_to_tmp[trg_offset+i] = tmp_offset+i
                for i in [-3,-2,-1]:
                    trg_to_tmp[trg_offset+trg_peplen+i] = tmp_offset+tmp_peplen+i
            trg_pmhc_seq = trg_mhc_seq + peptide
            tmp_pmhc_seq = tmp_mhc_seq + templatel.pep_seq
            identities = sum(trg_pmhc_seq[i] == tmp_pmhc_seq[j]
                             for i,j in trg_to_tmp.items())/len(trg_pmhc_seq)
            identities_for_sorting = (identities - 0.01*templatel.mhc_total_chainbreak)
            assert abs(identities_for_sorting-idents)<1e-3

            if verbose:
                print(f'oldnew mhc_idents: {idents:6.3f} {identities:6.3f} {pdbid}')
            show_alignment(trg_to_tmp, trg_pmhc_seq, tmp_pmhc_seq)
            pmhc_alignments.append((identities_for_sorting,
                                    pdbid,
                                    trg_to_tmp,
                                    trg_pmhc_seq,
                                    tmp_pmhc_seq,
                                    identities,
            ))
    else: # class II
        #
        trg_mhca_alseq = get_mhc_class_2_alseq('A', mhc_allele.split(',')[0])
        trg_mhcb_alseq = get_mhc_class_2_alseq('B', mhc_allele.split(',')[1])
        trg_mhca_seq = trg_mhca_alseq.replace(ALL_GENES_GAP_CHAR,'')
        trg_mhcb_seq = trg_mhcb_alseq.replace(ALL_GENES_GAP_CHAR,'')
        trg_pmhc_seq = trg_mhca_seq + trg_mhcb_seq + peptide

        sortl = []
        for l in ternary_info.itertuples():
            if (l.organism!=organism or l.mhc_class!=mhc_class or
                l.pdbid in BAD_PMHC_PDBIDS or l.pdbid in exclude_pdbids):
                continue
            mismatches_for_excluding = min(
                count_peptide_mismatches(x, l.pep_seq)
                for x in [peptide]+alt_self_peptides)

            if mismatches_for_excluding < min_pmhc_peptide_mismatches:
                if verbose:
                    print('peptide too close:', peptide, l.pep_seq,
                          mismatches_for_excluding, alt_self_peptides)
                continue
            tmp_mhca_alseq, tmp_mhcb_alseq = l.mhc_alignseq.split('/')
            idents = 0
            for a,b in zip([trg_mhca_alseq, trg_mhcb_alseq, peptide],
                           [tmp_mhca_alseq, tmp_mhcb_alseq, l.pep_seq]):
                assert len(a) == len(b)
                idents += sum(x==y for x,y in zip(a,b) if x!=ALL_GENES_GAP_CHAR)
            sortl.append((idents/len(trg_pmhc_seq), l.pdbid))
        sortl.sort(reverse=True)
        max_idents = sortl[0][0]
        print(f'mhc max_idents: {max_idents:.3f}', mhc_allele, peptide,
              ternary_info.loc[sortl[0][1], 'mhc_allele'],
              ternary_info.loc[sortl[0][1], 'pep_seq'])

        pmhc_alignments = []
        for (idents, pdbid) in sortl[:num_templates_per_run]:
            if idents < next_best_identity_threshold_tcr*max_idents:
                break
            templatel = ternary_info.loc[pdbid]
            tmp_mhca_alseq, tmp_mhcb_alseq = templatel.mhc_alignseq.split('/')
            mhca_part = tmp_mhca_alseq.replace(ALL_GENES_GAP_CHAR,'')
            mhcb_part = tmp_mhcb_alseq.replace(ALL_GENES_GAP_CHAR,'')
            tmp_mhca_seq, tmp_mhcb_seq = templatel.chainseq.split('/')[:2]
            assert mhca_part in tmp_mhca_seq and mhcb_part in tmp_mhcb_seq
            mhca_npad = tmp_mhca_seq.find(mhca_part)
            mhcb_npad = tmp_mhcb_seq.find(mhcb_part)

            trg_offset, tmp_offset = 0, mhca_npad
            al1 = {i-trg_mhca_alseq[:i].count(ALL_GENES_GAP_CHAR)+trg_offset:
                   i-tmp_mhca_alseq[:i].count(ALL_GENES_GAP_CHAR)+tmp_offset
                   for i,(a,b) in enumerate(zip(trg_mhca_alseq, tmp_mhca_alseq))
                   if a != ALL_GENES_GAP_CHAR and b != ALL_GENES_GAP_CHAR}
            trg_offset = len(trg_mhca_seq)
            tmp_offset = len(tmp_mhca_seq)+mhcb_npad
            al2 = {i-trg_mhcb_alseq[:i].count(ALL_GENES_GAP_CHAR)+trg_offset:
                   i-tmp_mhcb_alseq[:i].count(ALL_GENES_GAP_CHAR)+tmp_offset
                   for i,(a,b) in enumerate(zip(trg_mhcb_alseq, tmp_mhcb_alseq))
                   if a != ALL_GENES_GAP_CHAR and b != ALL_GENES_GAP_CHAR}
            trg_offset = len(trg_mhca_seq)+len(trg_mhcb_seq)
            tmp_offset = len(tmp_mhca_seq)+len(tmp_mhcb_seq)
            al3 = {i+trg_offset:i+tmp_offset for i in range(CLASS2_PEPLEN)}
            trg_to_tmp = {**al1, **al2, **al3}
            tmp_pmhc_seq = tmp_mhca_seq + tmp_mhcb_seq + templatel.pep_seq
            idents_redo = sum(trg_pmhc_seq[i] == tmp_pmhc_seq[j]
                              for i,j in trg_to_tmp.items())/len(trg_pmhc_seq)
            #print(f'oldnew mhc_idents: {idents:6.3f} {idents_redo:6.3f} {pdbid}')
            assert abs(idents-idents_redo)<1e-4
            show_alignment(trg_to_tmp, trg_pmhc_seq, tmp_pmhc_seq)
            pmhc_alignments.append((idents,
                                    pdbid,
                                    trg_to_tmp,
                                    trg_pmhc_seq,
                                    tmp_pmhc_seq,
                                    idents,
            ))





    tcr_alignments = {'A':[], 'B':[]}

    # compute single-chain tcrdists to candidate template chains

    # drop out of the loop if vdist hits this value and we've already got enough
    # templates
    big_v_dist=50

    for ab, trg_v, trg_j, trg_cdr3 in  [['A',va,ja,cdr3a],['B',vb,jb,cdr3b]]:
        trg_tcr = (organism[0]+trg_v, trg_j, trg_cdr3)

        trg_msa_alignments = align_vgene_to_structure_msas(organism, trg_v)

        templates = tcr_info[tcr_info.ab==ab]
        templates = templates[~templates.pdbid.isin(exclude_pdbids)]
        template_tcrs = [
            (x.organism[0]+x.v_gene, None, x.cdr3) for x in templates.itertuples()]
        #templates['v_gene j_gene cdr3'.split()].itertuples(index=False))
        closest_tcrs = [(x[0],x[1],trg_cdr3) for x in template_tcrs] # tmp v, trg cdr3
        sortl = sorted(
            [(tcrdister.single_chain_distance(trg_tcr,x),
              tcrdister.single_chain_distance(trg_tcr,y),
              i)
             for i,(x,y) in enumerate(zip(closest_tcrs, template_tcrs))])

        alignments = []
        for closest_dist, dist, ind in sortl:
            if dist < min_single_chain_tcrdist:
                #print('too close:', dist, trg_v, trg_j, trg_cdr3, template_tcrs[ind])
                continue
            if closest_dist > big_v_dist and len(alignments)>=num_templates_per_run:
                #print('too far:', closest_dist)
                break

            templatel = templates.iloc[ind]

            # if templatel.organism == 'human' and organism == 'human':
            #     msa_type = 'human'
            # else:
            #     msa_type = 'both'

            trg_to_tmp, trg_chainseq = align_vgene_to_template_pdb_chain(
                ab, organism, trg_v, trg_j, trg_cdr3, trg_msa_alignments,
                templatel.pdbid,
            )

            # tmp_chainseq_to_msa = align_tcr_info_pdb_chain_to_structure_msa(
            #     templatel.pdbid, templatel.ab, msa_type)

            identities = sum(trg_chainseq[i]==templatel.chainseq[j]
                             for i,j in trg_to_tmp.items()) / len(trg_chainseq)

            identities_for_sorting = identities + ternary_bonus * templatel.ternary

            alignments.append(
                (identities_for_sorting,
                 templatel.pdbid,
                 trg_to_tmp,
                 trg_chainseq,
                 templatel.chainseq,
                 closest_dist,
                 identities,
                ))
        alignments.sort(reverse=True)
        max_idents = alignments[0][0]
        tcr_alignments[ab] = [x for x in alignments[:num_templates_per_run]
                              if x[0] >= next_best_identity_threshold_tcr*max_idents]

    # docking geometries
    # exclude same-epitope geoms
    if pick_dgeoms_using_tcrdist:
        dgeom_info_by_chain = pick_dgeom_templates(
            num_templates_per_run, organism, mhc_class, mhc_allele,
            peptide, va, cdr3a, vb, cdr3b,
            min_peptide_mismatches = min_dgeom_peptide_mismatches,
            min_paired_tcrdist = min_dgeom_paired_tcrdist,
            min_singlechain_tcrdist = min_dgeom_singlechain_tcrdist,
            peptides_for_filtering = exclude_docking_geometry_peptides,
            exclude_pdbids = exclude_pdbids,
        )
    elif use_same_pmhc_dgeoms:
        # require matching organism, mhc, peptide with dgeoms
        dgeom_info = ternary_info[ternary_info.mhc_class == mhc_class]
        dgeom_info = dgeom_info[~dgeom_info.pdbid.isin(BAD_DGEOM_PDBIDS)]
        dgeom_info = dgeom_info[~dgeom_info.pdbid.isin(exclude_pdbids)]
        print('same mhc_class', dgeom_info.shape)
        dgeom_info = dgeom_info[dgeom_info.organism == organism]
        print('same organism', dgeom_info.shape)
        dgeom_info = dgeom_info[dgeom_info.mhc_allele == mhc_allele]
        print('same mhc_allele', dgeom_info.shape)
        dgeom_info = dgeom_info[dgeom_info.pep_seq == peptide]
        print('same peptide', dgeom_info.shape)
        if dgeom_info.shape[0]==0:
            print('ERROR no matching pmhc templates for dgeom:',
                  organism, mhc_class, mhc_allele, peptide)
            exit()
        dgeoms = [DockingGeometry().from_dict(x) for _,x in dgeom_info.iterrows()]
        if num_runs*num_templates_per_run > len(dgeoms):
            rep_dgeom_indices = np.random.permutation(len(dgeoms))
            rep_dgeoms = [dgeoms[x] for x in rep_dgeom_indices]
        else:
            dummy_organism = 'human' # just used for avg cdr coords
            rep_dgeoms, rep_dgeom_indices = docking_geometry.pick_docking_geometry_reps(
                dummy_organism, dgeoms, num_runs*num_templates_per_run)
    else:

        dgeom_info = ternary_info[ternary_info.mhc_class == mhc_class]
        if organism=='human': # restrict to human docks
            dgeom_info = dgeom_info[dgeom_info.organism == organism]
        dgeom_info = dgeom_info[~dgeom_info.pdbid.isin(BAD_DGEOM_PDBIDS)]
        dgeom_info = dgeom_info[~dgeom_info.pdbid.isin(exclude_pdbids)]
        dgeom_info = filter_templates_by_tcrdist(
            dgeom_info, organism, va, cdr3a, vb, cdr3b,
            min_paired_tcrdist=min_dgeom_paired_tcrdist,
            min_singlechain_tcrdist=min_dgeom_singlechain_tcrdist,
        )
        dgeom_info = filter_templates_by_peptide_mismatches(
            dgeom_info, organism, mhc_class, exclude_docking_geometry_peptides,
            min_dgeom_peptide_mismatches)
        dgeoms = [DockingGeometry().from_dict(x) for _,x in dgeom_info.iterrows()]

        if num_runs*num_templates_per_run > len(dgeoms):
            rep_dgeom_indices = np.random.permutation(len(dgeoms))
            rep_dgeoms = [dgeoms[x] for x in rep_dgeom_indices]
        else:
            dummy_organism = 'human' # just used for avg cdr coords
            rep_dgeoms, rep_dgeom_indices = docking_geometry.pick_docking_geometry_reps(
                dummy_organism, dgeoms, num_runs*num_templates_per_run)

    #print('dgeoms:', len(dgeoms), len(rep_dgeoms))

    trg_pmhc_seq = pmhc_alignments[0][3] # check for consistency below
    trg_tcra_seq = tcr_alignments['A'][0][3] # ditto
    trg_tcrb_seq = tcr_alignments['B'][0][3] # ditto

    # now make the template pdbs

    dfl = []
    for run in range(num_runs):
        for itmp in range(num_templates_per_run):
            pmhc_al = pmhc_alignments[itmp%len(pmhc_alignments)]
            tcra_al = tcr_alignments['A'][itmp%len(tcr_alignments['A'])]
            tcrb_al = tcr_alignments['B'][itmp%len(tcr_alignments['B'])]
            if pick_dgeoms_using_tcrdist:
                chain = ['AB','A','B'][run]
                dgeom_info = dgeom_info_by_chain[chain]
                dgeom_row = dgeom_info.iloc[itmp%dgeom_info.shape[0]]
                dgeom = DockingGeometry().from_dict(dgeom_row)
            else:
                # order them this way so each run has diverse dgeoms...
                dgeom_repno = (itmp*num_runs + run)%len(rep_dgeoms)
                dgeom = rep_dgeoms[dgeom_repno]
                dgeom_row = dgeom_info.iloc[rep_dgeom_indices[dgeom_repno]]

            pmhc_pdbid = pmhc_al[1]
            pmhc_pose, pmhc_tdinfo = get_template_pose_and_tdinfo(pmhc_pdbid, PMHC)

            tcra_pdbid = tcra_al[1]
            tcra_pose, tcra_tdinfo = get_template_pose_and_tdinfo(tcra_pdbid, TCR)

            tcrb_pdbid = tcrb_al[1]
            tcrb_pose, tcrb_tdinfo = get_template_pose_and_tdinfo(tcrb_pdbid, TCR)

            # assert ((mhc_class==1 and pmhc_pose.num_chains() == 4) or
            #         (mhc_class==2 and pmhc_pose.num_chains() == 5))
            assert len(tcra_pose['chains']) == 2 and len(tcrb_pose['chains']) == 2

            # copy tcrb into tcra_pose by superimposing core coords
            fix_coords = tcra_pose['ca_coords'][tcra_tdinfo.tcr_core[core_len:]]
            mov_coords = tcrb_pose['ca_coords'][tcrb_tdinfo.tcr_core[core_len:]]
            R, v = superimpose.superimposition_transform(
                fix_coords, mov_coords)
            tcrb_pose = apply_transform_Rx_plus_v(tcrb_pose, R, v)
            tcra_pose = delete_chains(tcra_pose, [1])
            tcra_pose = append_chains(tcra_pose, tcrb_pose, [1])

            # update tcra_tdinfo, compute tcr_stub
            # use dgeom to compute desired tcr_stub
            # transform tcra_pose
            offset = tcra_pose['chainbounds'][1]-tcrb_pose['chainbounds'][1]
            tcra_tdinfo.tcr_core = (
                tcra_tdinfo.tcr_core[:core_len] +
                [x+offset for x in tcrb_tdinfo.tcr_core[core_len:]])
            tcra_tdinfo.tcr_cdrs = (
                tcra_tdinfo.tcr_cdrs[:4] +
                [[x+offset,y+offset] for x,y in tcrb_tdinfo.tcr_cdrs[4:]])

            assert tcra_pose['sequence'][tcra_tdinfo.tcr_cdrs[3][0]]=='C'
            assert tcra_pose['sequence'][tcra_tdinfo.tcr_cdrs[7][0]]=='C'
            old_tcr_stub = tcr_util.get_tcr_stub(tcra_pose, tcra_tdinfo)
            new_tcr_stub = docking_geometry.stub_from_docking_geometry(
                dgeom)
            # R @ old_tcr_stub['axes'].T = new_tcr_stub['axes'].T
            R = new_tcr_stub['axes'].T @ old_tcr_stub['axes']
            # R @ old_tcr_stub['origin'] + v = new_tcr_stub['origin']
            v = new_tcr_stub['origin'] - R@old_tcr_stub['origin']
            tcra_pose = apply_transform_Rx_plus_v(tcra_pose, R, v)

            #
            # copy tcr from tcra_pose into pmhc_pose
            num_pmhc_chains = mhc_class+1
            if len(pmhc_pose['chains']) > num_pmhc_chains:
                del_chains = list(range(num_pmhc_chains, len(pmhc_pose['chains'])))
                pmhc_pose = delete_chains(pmhc_pose, del_chains)
            pmhc_pose = append_chains(pmhc_pose, tcra_pose, [0,1])
            assert len(pmhc_pose['chains'])==2+num_pmhc_chains
            offset = pmhc_pose['chainbounds'][num_pmhc_chains]
            pmhc_tdinfo.tcr_core = (
                [x+offset for x in tcra_tdinfo.tcr_core])
            pmhc_tdinfo.tcr_cdrs = (
                [[x+offset,y+offset] for x,y in tcra_tdinfo.tcr_cdrs])
            assert pmhc_pose['sequence'][pmhc_tdinfo.tcr_cdrs[3][0]] == 'C'
            assert pmhc_pose['sequence'][pmhc_tdinfo.tcr_cdrs[7][0]] == 'C'

            # should be the same as new_tcr_stub!
            redo_tcr_stub = tcr_util.get_tcr_stub(pmhc_pose, pmhc_tdinfo)
            v_dev = norm(redo_tcr_stub['origin']-new_tcr_stub['origin'])
            M_dev = norm(new_tcr_stub['axes'] @ redo_tcr_stub['axes'].T - np.eye(3))
            if max(v_dev, M_dev)>5e-2:
                print('devs:', v_dev, M_dev)
            assert v_dev<5e-2
            assert M_dev<5e-2

            # setup the alignment
            trg_to_tmp = dict(pmhc_al[2])
            tmp_pmhc_seq = pmhc_al[4]
            tmp_tcra_seq = tcra_al[4]
            tmp_tcrb_seq = tcrb_al[4]
            tmp_fullseq = tmp_pmhc_seq + tmp_tcra_seq + tmp_tcrb_seq
            assert pmhc_pose['sequence'] == tmp_fullseq

            assert len(pmhc_pose['chains']) == num_pmhc_chains+2

            assert trg_pmhc_seq == pmhc_al[3]
            assert trg_tcra_seq == tcra_al[3]
            assert trg_tcrb_seq == tcrb_al[3]

            trg_fullseq = trg_pmhc_seq + trg_tcra_seq + trg_tcrb_seq
            trg_offset = len(trg_pmhc_seq)
            tmp_offset = len(tmp_pmhc_seq)
            trg_to_tmp.update({i+trg_offset:j+tmp_offset
                               for i,j in tcra_al[2].items()})
            trg_offset += len(trg_tcra_seq)
            tmp_offset += len(tmp_tcra_seq)
            trg_to_tmp.update({i+trg_offset:j+tmp_offset
                               for i,j in tcrb_al[2].items()})
            assert Counter(trg_to_tmp.values()).most_common(1)[0][1]==1

            identities = sum(trg_fullseq[i]==tmp_fullseq[j]
                             for i,j in trg_to_tmp.items())
            overall_idents = identities/len(trg_fullseq)
            if run==0:
                print(f'identities: {itmp} {overall_idents:.3f} '
                      f'pmhc: {pmhc_al[-1]:.3f} '
                      f'tcra: {tcra_al[-1]:.3f} '
                      f'tcrb: {tcrb_al[-1]:.3f} ', va, ja, cdr3a, vb, jb, cdr3b,
                      flush=True)
                if verbose:
                    for i,j in sorted(trg_to_tmp.items()):
                        a,b = trg_fullseq[i], tmp_fullseq[j]
                        star = '*' if a==b else ' '
                        print(f'{i:4d} {j:4d} {a} {star} {b}')

            outpdbfile = f'{outfile_prefix}_{run}_{itmp}.pdb'
            #pmhc_pose.dump_pdb(outpdbfile)
            dump_pdb(pmhc_pose, outpdbfile)
            #print('made:', outpdbfile)

            trg_pmhc_seqs = ([trg_mhc_seq, peptide] if mhc_class==1 else
                             [trg_mhca_seq, trg_mhcb_seq, peptide])
            trg_cbseq = '/'.join(trg_pmhc_seqs+[trg_tcra_seq, trg_tcrb_seq])

            alignstring = ';'.join(f'{i}:{j}' for i,j in trg_to_tmp.items())

            if pmhc_pdbid in pmhc_info.index:
                pmhc_allele=pmhc_info.loc[pmhc_pdbid, 'mhc_allele']
            else:
                pmhc_allele=ternary_info.loc[pmhc_pdbid, 'mhc_allele']

            outl = OrderedDict(
                run=run,
                template_no=itmp,
                target_chainseq=trg_cbseq,
                overall_idents=overall_idents,
                pmhc_pdbid=pmhc_pdbid,
                pmhc_idents=pmhc_al[-1],
                pmhc_allele=pmhc_allele,#pmhc_info.loc[pmhc_pdbid, 'mhc_allele'],
                tcra_pdbid=tcra_pdbid,
                tcra_idents=tcra_al[-1],
                tcra_v   =tcr_info.loc[(tcra_pdbid,'A'), 'v_gene'],
                tcra_j   =tcr_info.loc[(tcra_pdbid,'A'), 'j_gene'],
                tcra_cdr3=tcr_info.loc[(tcra_pdbid,'A'), 'cdr3'],
                tcrb_pdbid=tcrb_pdbid,
                tcrb_idents=tcrb_al[-1],
                tcrb_v   =tcr_info.loc[(tcrb_pdbid,'B'), 'v_gene'],
                tcrb_j   =tcr_info.loc[(tcrb_pdbid,'B'), 'j_gene'],
                tcrb_cdr3=tcr_info.loc[(tcrb_pdbid,'B'), 'cdr3'],
                dgeom_pdbid=dgeom_row.pdbid,
                template_pdbfile=outpdbfile,
                target_to_template_alignstring=alignstring,
                identities=identities,
                target_len=len(trg_fullseq),
                template_len=len(tmp_fullseq),
            )
            dfl.append(outl)
    assert len(dfl) == num_runs * num_templates_per_run
    return pd.DataFrame(dfl)

def genes_ok_for_modeling(organism, va, ja, vb, jb, verbose=True):

    if organism not in all_genes:
        print(f'ERROR unrecognized organism: "{organism}" expected one of',
              all_genes.keys())
        return False

    for g in [va, ja, vb, jb]:
        if g not in all_genes[organism]:
            from .tcrdist.basic import path_to_db, db_file
            print(f'ERROR: unrecognized gene: "{g}" for organism "{organism}"')
            # for error output:
            dbfile = path_to_db / db_file
            print('check the ids in', dbfile, 'for organism', organism)
            return False

    if (organism,va) not in both_structure_alignments.index:
        if verbose:
            print('ERROR: no va alignment:', organism, va)
        return False

    if (organism,vb) not in both_structure_alignments.index:
        if verbose:
            print('ERROR: no vb alignment:', organism, vb)
        return False

    va_seq = get_v_seq_up_to_cys(organism, va)
    ja_seq = get_j_seq_after_cdr3(organism, ja)
    vb_seq = get_v_seq_up_to_cys(organism, vb)
    jb_seq = get_j_seq_after_cdr3(organism, jb)

    if ( '*' in va_seq+ja_seq+vb_seq+jb_seq or
         va_seq[-1] != 'C' or vb_seq[-1] != 'C'):
        if verbose:
            print('ERROR bad seqs:', va, va_seq, ja, ja_seq,
                  vb, vb_seq, ja, jb_seq)
        return False
    return True

def check_genes_for_modeling(tcr_db):
    all_ok = True
    for index, targetl in tcr_db.iterrows():
        ok = genes_ok_for_modeling(
            targetl.organism, targetl.va, targetl.ja, targetl.vb, targetl.jb,
            verbose=True)
        all_ok = all_ok and ok
        if not ok:
            print('ERROR bad tcr genes at index=', index)
    return all_ok


def setup_for_alphafold(
        tcr_db,
        outdir,
        organism=None, # if None, should be present in tcr_db
        min_pmhc_count=None,
        max_pmhc_count=None,
        random_seed=1, # for subsampling tcrs
        num_runs=3,
        clobber=False,
        exclude_self_peptide_docking_geometries=False,
        alt_self_peptides_column=None, # this column should be comma-separated
        exclude_pdbids_column=None, # this column should be comma-separated
        targetid_prefix_suffix='',
        **kwargs,
):
    '''
    '''
    assert outdir.endswith('/')
    required_cols = 'va ja cdr3a vb jb cdr3b mhc_class mhc peptide'.split()
    if organism is None:
        required_cols.append('organism')
    for col in required_cols:
        assert col in tcr_db.columns

    #assert not any(tcr_db.mhc.str.startswith('E*'))

    tcr_db = tcr_db.copy()
    tcr_db['mhc_peptide'] = (tcr_db.mhc.str.replace('*','').str.replace(':','')+
                             '_'+tcr_db.peptide)
    if organism is not None:
        if 'organism' in tcr_db.columns:
            assert all(tcr_db.organism==organism)
        else:
            tcr_db['organism'] = organism

    # doesnt do anything if init has already been called (I don't think)

    if not exists(outdir):
        os.mkdir(outdir)
    else:
        assert clobber, 'outdir already exists: '+outdir

    #optionally filter to peptide mhc combos with min/max counts ##########
    if min_pmhc_count is not None or max_pmhc_count is not None:
        tcr_db.sort_values('mhc_peptide', inplace=True) ## IMPORTANT B/C MASKING
        random.seed(random_seed) # shuffling of epitope tcrs
        assert min_pmhc_count is not None and max_pmhc_count is not None
        mhc_peptides = tcr_db.mhc_peptide.drop_duplicates().to_list()
        mask = []
        for mhc_peptide in mhc_peptides:
            count = np.sum(tcr_db.mhc_peptide==mhc_peptide)
            if count < min_pmhc_count:
                mhc_peptide_mask = [False]*count
            elif count <= max_pmhc_count:
                mhc_peptide_mask = [True]*count
            else:
                mhc_peptide_mask = [True]*max_pmhc_count+ [False]*(count-max_pmhc_count)
                random.shuffle(mhc_peptide_mask)
            mask.extend(mhc_peptide_mask)

        old_size = tcr_db.shape[0]
        tcr_db = tcr_db[mask].copy()
        print('subset tcr_db:', old_size, tcr_db.shape[0])
        counts = tcr_db.mhc_peptide.value_counts().to_list()
        assert min_pmhc_count <= min(counts) and max(counts) <= max_pmhc_count

    tcr_db_outfile = outdir+'tcr_db.tsv'
    tcr_db.to_csv(tcr_db_outfile, sep='\t', index=False)

    tcr_db = tcr_db.reset_index(drop=True)

    print('check genes for modeling', tcr_db.shape[0])
    for index, targetl in tcr_db.iterrows():
        if not genes_ok_for_modeling(
                targetl.organism, targetl.va, targetl.ja, targetl.vb, targetl.jb):
            print('ERROR bad genes:', index, targetl.va, targetl.ja,
                  targetl.vb, targetl.jb)
            assert False
            exit()

    targets_dfl = []
    for index, targetl in tcr_db.iterrows():
        targetid_prefix = f'T{index:05d}_{targetl.mhc_peptide}{targetid_prefix_suffix}'
        print('START', index, tcr_db.shape[0], targetid_prefix)
        outfile_prefix = f'{outdir}{targetid_prefix}'
        if exclude_self_peptide_docking_geometries:
            exclude_docking_geometry_peptides = [targetl.peptide]
        else:
            exclude_docking_geometry_peptides = []
        if alt_self_peptides_column is not None:
            assert exclude_docking_geometry_peptides
            alt_self_peptides = targetl[alt_self_peptides_column].split(',')
            if exclude_self_peptide_docking_geometries:
                exclude_docking_geometry_peptides.extend(alt_self_peptides)
        else:
            alt_self_peptides=None

        if exclude_pdbids_column is not None:
            exclude_pdbids = targetl[exclude_pdbids_column]
            if pd.isna(exclude_pdbids):
                exclude_pdbids = None
            else:
                exclude_pdbids = exclude_pdbids.split(',')
        else:
            exclude_pdbids = None

        all_run_info = make_templates_for_alphafold(
            targetl.organism, targetl.va, targetl.ja, targetl.cdr3a,
            targetl.vb, targetl.jb, targetl.cdr3b,
            targetl.mhc_class, targetl.mhc, targetl.peptide, outfile_prefix,
            exclude_docking_geometry_peptides=exclude_docking_geometry_peptides,
            num_runs = num_runs,
            alt_self_peptides=alt_self_peptides,
            exclude_pdbids = exclude_pdbids,
            **kwargs,
        )

        for run in range(num_runs):
            info = all_run_info[all_run_info.run==run]
            assert info.shape[0] == 4#num templates
            targetid = f'{targetid_prefix}_{run}'
            trg_cbseq = set(info.target_chainseq).pop()
            alignfile = f'{outdir}{targetid}_alignments.tsv'
            info.to_csv(alignfile, sep='\t', index=False)
            outl = pd.Series(targetl)
            outl['targetid'] = targetid
            outl['target_chainseq'] = trg_cbseq
            outl['templates_alignfile'] = alignfile
            targets_dfl.append(outl)
        sys.stdout.flush()

        # save partial work... since this is so freakin slow
        outfile = outdir+'targets.tsv'
        pd.DataFrame(targets_dfl).to_csv(outfile, sep='\t', index=False)

    outfile = outdir+'targets.tsv'
    pd.DataFrame(targets_dfl).to_csv(outfile, sep='\t', index=False)
    print('made:', outfile)


def get_mhc_chain_trim_positions(chainseq, organism, mhc_class, mhc_allele, chain=None):
    '''
    '''
    assert mhc_class in [1,2]
    if mhc_class==2:
        assert chain in ['A','B']
        alseq = get_mhc_class_2_alseq(chain, mhc_allele)
        if alseq is None:
            print('ERROR: None alseq:', organism, mhc_class, mhc_allele)
            return [None]
        if chain == 'A':
            #H2AKa EPQGGLQNIATGKHNLEI
            #DRA   EAQGALANIAVDKANLEI
            ref_alseq = 'HVIIQ.AEFYLNPDQSGEFMFDFDGDEIFHVDMAKKETVWRLEEFGRFASFEAQGALANIAVDKANLEIMTKRSN'
            ref_helices = ['EAQGALANIAVDKANLEI']
        else:
            #H2AKb   YWNKQ..YLERTRAELDTVCRHN
            #DRB1*01 YWNSQKDLLEQRRAAVDTYCRHN
            ref_alseq = 'FVHQFQPFCYFTNGTQRIRLVIRYIYNREEYVRFDSDVGEYRAVTELGRPDAEYWNKQ..YLERTRAELDTVCRHNYEKTETPTS'
            ref_helices = ['YWNKQ..YLERTRAELDTVCRHN']

    else:
        alseq = get_mhc_class_1_alseq(mhc_allele)
        if alseq is None:
            print('ERROR: None alseq:', organism, mhc_class, mhc_allele)
            return [None]
        #A02: GETRKVKAHSQTHRVDLGT and KWEAAHVAEQLRAYLEGTCVEW
        #D-B: RETQKAKGQEQWFRVSLRN and KWEQSGAAEHYKAYLEGECVEW
        if organism=='mouse':
            ref_alseq = 'GPHSMRYFETAVSRPGLEEPRYISVGYVDNKEFVRFDSDAENPRYEPRAPWMEQEGPEYWERETQKAKGQEQWFRVSLRNLLGYYNQSAGGSHTLQQMSGCDLGSDWRLLRGYLQFAYEGRDYIALNEDLKTWTAADMAAQITRRKWEQSGAAEHYKAYLEGECVEWLHRYLKNGNATLLR'
            ref_helices = ['RETQKAKGQEQWFRVSLRN', 'KWEQSGAAEHYKAYLEGECVEW']
        else:
            ref_alseq = 'GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENG'
            ref_helices = ['GETRKVKAHSQTHRVDLGT', 'KWEAAHVAEQLRAYLEGTCVEW']

    assert len(alseq) == len(ref_alseq)
    alseqseq = alseq.replace(ALL_GENES_GAP_CHAR, '')
    if chainseq != alseqseq:
        #print('WARNING: mhc seq mismatch: chainseq=', chainseq)
        #print('WARNING: mhc seq mismatch: alseqseq=', alseqseq)
        as2cs = blosum_align(alseqseq, chainseq)
    else:
        as2cs = {i:i for i in range(len(chainseq))}

    positions = []
    for helix in ref_helices:
        start = ref_alseq.index(helix)
        for pos in range(start, start+len(helix)):
            if alseq[pos] != ALL_GENES_GAP_CHAR:
                i = pos-alseq[:pos].count(ALL_GENES_GAP_CHAR)
                if i in as2cs:
                    positions.append(as2cs[i])
    return positions
