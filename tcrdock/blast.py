import random
import pandas as pd
from pathlib import Path
from os import system, remove
from os.path import exists, isdir
from .util import amino_acids

path_to_blast_executables = Path(__file__).parents[1] / 'ncbi-blast-2.11.0+' / 'bin'
assert isdir( path_to_blast_executables ),\
    'You need to download blast; please run download_blast.py in TCRdock/ folder'

# used this command to make a blast database:
#~/bin/blastplus/bin/makeblastdb -in mhc_pdb_chains_mouse_reps_reformat.fasta -dbtype prot

blastp_exe = str(path_to_blast_executables / 'blastp')

blast_fields = ('evalue bitscore qaccver saccver pident length mismatch'
                ' gapopen qstart qend qlen qseq sstart send slen sseq')


def blast_sequence_and_read_hits(
        query_sequence,
        dbfile,
        tmpfile_prefix = '',
        evalue = 1e-3,
        num_alignments = 10000,
        verbose=False,
):
    tmpfile = f'{tmpfile_prefix}tmp_fasta_{random.random()}.fasta'
    out = open(tmpfile, 'w')
    out.write(f'>tmp\n{query_sequence}\n')
    out.close()

    outfile = tmpfile+'.blast'

    cmd = (f'{blastp_exe} -query {tmpfile} -db {dbfile}'
           f' -outfmt "10 delim=, {blast_fields}" -evalue {evalue}'
           f' -num_alignments {num_alignments} -out {outfile}')

    if not verbose:
        cmd += ' 2> /dev/null'

    if verbose:
        print(cmd)
    system(cmd)

    blast_hits = pd.read_csv(
        outfile, header=None, names=blast_fields.split())
    #blast_hits.rename(columns={'saccver':'pdb_chain', 'qaccver':'allele'}, inplace = True)
    #blast_hits.sort_values('pident', ascending=False, inplace=True)

    for filename in [tmpfile, outfile]:
        if exists(filename):
            remove(filename)

    return blast_hits



def setup_query_to_hit_map(hit):
    ''' hit is a single row from blast_hits

    query2hit_align is a dictionary mapping from full-length, 0-indexed positions
    in query sequence to full-length, 0-indexed positions in hit sequence
    '''

    query2hit_align = {}
    for ii,(qaa,haa) in enumerate(zip(hit.qseq, hit.sseq)):
        if qaa in amino_acids and haa in amino_acids:
            qpos = hit.qstart + ii - hit.qseq[:ii].count('-') - 1 #0-idx
            hpos = hit.sstart + ii - hit.sseq[:ii].count('-') - 1 #
            query2hit_align[qpos] = hpos

    return query2hit_align
