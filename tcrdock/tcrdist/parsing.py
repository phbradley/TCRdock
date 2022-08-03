######################################################################################88

import sys
import random
from collections import OrderedDict, Counter
import pandas as pd
from .basic import path_to_db
from os import system
from os.path import exists
from . import translation
from .all_genes import all_genes, gap_character
from .genetic_code import genetic_code, reverse_genetic_code
from . import logo_tools
from ..blast import (blast_sequence_and_read_hits, setup_query_to_hit_map,
                     path_to_blast_executables)

def get_blast_db_path(organism, ab, vj):
    ''' This is the protein sequence db
    '''
    return path_to_db / f'imgt_prot_blast_db_{organism}_{ab}_{vj}.fasta'


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

        if False:# helpful for debugging/info
            for v,g in all_genes[organism].items():
                allele = int(v.split('*')[-1]) # sanity check allele names end with *int
                if g.chain == ab and g.region == 'V' and v.endswith('*01'):
                    coreseq = ''.join([g.alseq[x] for x in positions])
                    if '*' in coreseq or coreseq[1]+coreseq[-1] != 'CC':
                        print('funny coreseq:', coreseq, v, organism, ab)

        all_core_alseq_positions_0x[(organism,ab)] = positions


def get_core_positions_0x(organism, v_gene):
    vg = all_genes[organism][v_gene]

    alseq = vg.alseq

    core_positions_0x = []
    for pos in all_core_alseq_positions_0x[(organism, vg.chain)]:
        core_positions_0x.append(pos - alseq[:pos].count(gap_character))
    return core_positions_0x

def make_blast_dbs():
    makeblastdb_exe = str(path_to_blast_executables / 'makeblastdb')

    # protein sequences
    for organism in all_genes:
        for ab in 'AB':
            for vj in 'VJ':
                pname = get_blast_db_path(organism, ab, vj)
                out = open(pname, 'w')
                for id, g in all_genes[organism].items():
                    if g.chain == ab and g.region == vj:
                        out.write(f'>{id}\n{g.protseq}\n')
                out.close()
                print('made:', pname)

                # format the db
                cmd = f'{makeblastdb_exe} -in {pname} -dbtype prot'
                print(cmd)
                system(cmd)




def parse_core_positions( organism, ab, qseq, v_gene, q2v_align ):
    ''' returns qcore_positions, mismatches

    qcore_positions is a list of 0-indexed core positions in qseq, may contain -1 if
    the q2v_align alignment does not cover all the v_gene core positions
    '''
    vseq = all_genes[organism][v_gene].protseq

    core_positions = get_core_positions_0x(organism, v_gene)

    qcore_positions = [-1]*len(core_positions)
    mismatches = 0

    for qpos, vpos in q2v_align.items():
        if vpos in core_positions:
            qcore_positions[core_positions.index(vpos)] = qpos
            if qseq[qpos] != vseq[vpos]:
                mismatches += 1

    return qcore_positions, mismatches


def parse_other_cdrs(organism, ab, qseq, v_gene, q2v_align):
    v_seq = all_genes[organism][v_gene].protseq
    alseq = all_genes[organism][v_gene].alseq
    cdr_cols_1x = all_genes[organism][v_gene].cdr_columns

    assert alseq.replace(gap_character,'') == v_seq
    alseq2seq = {i:i-alseq[:i].count(gap_character)
                 for i,a in enumerate(alseq)
                 if a != gap_character}

    assert len(cdr_cols_1x) == 4 ## CDR1, CDR2, CDR2.5, and CDR3(start)

    qseq_cdr_bounds = []
    cdr_mismatches = 0
    for start_col_1x, stop_col_1x in cdr_cols_1x[:3]:
        start = alseq2seq[start_col_1x-1]
        stop  = alseq2seq[stop_col_1x-1]

        ## what aligns to this region in v_hit
        q_loop_positions = [i for i,j in q2v_align.items() if start <= j <= stop]
        if q_loop_positions:
            qstart = min( q_loop_positions )
            qstop  = max( q_loop_positions )
            qseq_cdr_bounds.append([qstart, qstop])
            for i in range(qstart,qstop+1):
                if i not in q2v_align or v_seq[q2v_align[i]] != qseq[i]:
                    cdr_mismatches += 1
        else:
            qseq_cdr_bounds.append((None,None))
            cdr_mismatches += stop-start+1
    return qseq_cdr_bounds, cdr_mismatches


def parse_cdr3(organism, ab, qseq, v_gene, j_gene, q2v_align, q2j_align):
    ''' returns (C position, F position) ie (CDR3 start, CDR3 stop)
    in 0-indexed qseq numbering
    either will be None if parsing failed
    '''
    vg = all_genes[organism][v_gene]
    jg = all_genes[organism][j_gene]

    ## what is the C position in this v gene?
    cpos = vg.cdr_columns[3][0]-1 ## now 0-indexed, vg.alseq numbers
    cpos -= vg.alseq[:cpos].count(gap_character) # now vg.protseq numbers

    v2q_align = {j:i for i,j in q2v_align.items()}
    query_cpos = v2q_align.get(cpos, None)

    # V mismatches outside cdr3 region
    v_mismatches = sum(qseq[i] != vg.protseq[j]
                       for i,j in q2v_align.items()
                       if j < cpos)

    fpos = jg.cdr_columns[0][1]-1 # 0-idx jg.alseq numbers
    fpos -= jg.alseq[:fpos].count(gap_character) # 0-idx, jg.protseq numbers

    j2q_align = {j:i for i,j in q2j_align.items()}
    query_fpos = j2q_align.get(fpos, None)

    j_mismatches = sum(qseq[i] != jg.protseq[j]
                       for i,j in q2j_align.items()
                       if j > fpos)

    return [query_cpos, query_fpos], v_mismatches, j_mismatches


def get_top_blast_hit_with_allele_sorting(hits):
    ''' In the case of ties, prefer lower allele numbers
    '''
    top_bitscore = max(hits.bitscore)
    top_hits = hits[hits.bitscore==top_bitscore].copy()
    #print('top_bitscore:', top_bitscore, 'ties:', top_hits.shape[0])
    top_hits['allele'] = top_hits.saccver.str.split('*').str.get(-1).astype(int)
    return top_hits.sort_values('allele').iloc[0].copy()

def parse_tcr_sequence(organism, chain, sequence):
    ''' return dict with keys
    * cdr_loops
    * core_positions
    * v_gene
    * j_gene
    * mismatches (which is a dict with keys=regions and values=ints)

    returns empty dict on failure

    cdr_loops = [[start1,stop1], [start2,stop2], [start2.5, stop2.5]]
    core_positions = [pos0, pos1, ..., pos12]

    all numbers are 0-indexed with respect to chainseq

    returns None upon parse failure

    '''
    assert chain in 'AB'

    tmpdbfile = get_blast_db_path(organism, chain, 'V')
    if not exists(tmpdbfile):
        make_blast_dbs() # only need to do this once
        assert exists(tmpdbfile)

    top_hits = []
    for vj in 'VJ':
        dbfile = get_blast_db_path(organism, chain, vj)
        hits = blast_sequence_and_read_hits(sequence, dbfile)

        if hits.shape[0]:
            top_hits.append(get_top_blast_hit_with_allele_sorting(hits))

    if len(top_hits) == 2:
        v_hit, j_hit = top_hits

        q2v_align = setup_query_to_hit_map(v_hit)
        q2j_align = setup_query_to_hit_map(j_hit)

        v_gene = v_hit.saccver
        j_gene = j_hit.saccver

        cdr3_bounds, v_mismatches, j_mismatches = parse_cdr3(
            organism, chain, sequence, v_gene, j_gene, q2v_align, q2j_align,
        )

        other_cdr_bounds, other_cdr_mismatches = parse_other_cdrs(
            organism, chain, sequence, v_gene, q2v_align,
        )

        core_positions, core_mismatches = parse_core_positions(
            organism, chain, sequence, v_gene, q2v_align
        )

        if -1 in core_positions:
            # fail
            return {}

        cdr_loops = other_cdr_bounds + [cdr3_bounds]
        if any(None in bounds for bounds in cdr_loops):
            return {} # signal failure ### EARLY RETURN!!!

        result = {
            'cdr_loops':cdr_loops,
            'core_positions':core_positions,
            'v_gene': v_gene,
            'j_gene': j_gene,
            'mismatches':{
                'v_mismatches':v_mismatches,
                'j_mismatches':j_mismatches,
                'other_cdr_mismatches':other_cdr_mismatches,
                'core_mismatches':core_mismatches,
            },
        }
        return result
    else:
        print('failed to find v and j matches')
        return None
