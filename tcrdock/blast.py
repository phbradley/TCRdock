import random
import pandas as pd
from pathlib import Path
from os import system, remove
from os.path import exists, isdir
from .util import amino_acids

path_to_blast_executables = Path(__file__).parents[1] / 'ncbi-blast-2.11.0+' / 'bin'
assert isdir( path_to_blast_executables ),\
    'You need to download blast; please run download_blast.py in TCRdock/ folder'

blastp_exe = str(path_to_blast_executables / 'blastp')
makeblastdb_exe = str(path_to_blast_executables / 'makeblastdb')

blast_fields = ('evalue bitscore qaccver saccver pident length mismatch'
                ' gapopen qstart qend qlen qseq sstart send slen sseq')

def make_blast_dbs(fastafile, dbtype='prot'):
    assert dbtype in ['prot','nucl']

    cmd = f'{makeblastdb_exe} -in {fastafile} -dbtype {dbtype}'
    print(cmd)
    system(cmd)


def check_for_blast_dbs(fastafile):
    return exists(str(fastafile)+'.phr')

def blast_file_and_read_hits(
        fname,
        dbfile,
        evalue = 1e-3,
        num_alignments = 10000,
        verbose=False,
        clobber=False,
        extra_blast_args=''
):
    assert exists(dbfile), f'missing file for BLAST-ing against: {dbfile}'

    # check for blast database files
    if not check_for_blast_dbs(dbfile):
        # maybe we haven't set up the blast files yet...
        print('WARNING: missing blast db files, trying to create...')
        make_blast_dbs(dbfile)
        assert check_for_blast_dbs(dbfile), 'Failed to create blast db files!'

    outfile = fname+'.blast'
    assert clobber or not exists(outfile)

    cmd = (f'{blastp_exe} -query {fname} -db {dbfile} {extra_blast_args} '
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

    if exists(outfile):
        remove(outfile)

    return blast_hits


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

    blast_hits = blast_file_and_read_hits(
        tmpfile, dbfile, evalue, num_alignments, verbose, clobber=True)

    if exists(tmpfile):
        remove(tmpfile)

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
