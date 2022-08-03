# import numpy as np
# import sys
# from os import system
import os.path
from os import system, popen
from os.path import exists
from pathlib import Path
import pandas as pd
from .util import amino_acids, standard_three_letter_codes, path_to_db

def get_pdbfile(pdbid, desired_filename, verbose=False, clobber=False):
    if exists(desired_filename) and not clobber:
        print('tcrdock.pdb_util.get_pdbfile:: file already exists:',
              desired_filename,'set clobber=True to overwrite')
        return
    pdbid = pdbid.lower()
    cmd = f'wget ftp://ftp.wwpdb.org/pub/pdb/data/structures/divided/pdb/{pdbid[1:3]}/pdb{pdbid}.ent.gz'
    if verbose:
        print(cmd)
    system(cmd)
    filename = 'pdb%s.ent.gz'%pdbid
    if not exists( filename ):
        print('ERROR get_pdbfile FAILED:', pdbid)
        return None
    assert exists( filename )
    system('gunzip -f '+filename) # force overwrite if already exists
    filename = filename[:-3]
    assert exists( filename )
    if filename != desired_filename:
        system(f'mv {filename} {desired_filename}')
    assert exists( desired_filename )
    return desired_filename

pdb_protein_three_letter_codes = None
def get_pdb_protein_three_letter_codes():
    global pdb_protein_three_letter_codes
    if pdb_protein_three_letter_codes is not None:
        return pdb_protein_three_letter_codes

    tsvfile = Path.joinpath( path_to_db, 'pdb_codes.tsv')
    df = pd.read_csv(tsvfile, sep='\t')

    # restrict to 3letter codes that have a valid 1letter code (not sure if this really matters at all!)
    mask = [x.code1 in amino_acids for x in df.itertuples()]
    df = df[mask]
    pdb_protein_three_letter_codes = set(df.code3)
    pdb_protein_three_letter_codes.update(standard_three_letter_codes)

    return pdb_protein_three_letter_codes


def clean_pdbfile(oldfile, newfile, clobber=False):
    print('making:', newfile)
    if exists(newfile):
        if not clobber:
            print(f'ERROR clean_pdbfile would be overwriting {newfile}')
            print('please remove it or pass clobber=True')
            return

    keepers = ['REMARK   2 RESOL', 'CRYST1']

    good_rsd_names = get_pdb_protein_three_letter_codes() # defined above

    out = open(newfile,'w')
    with open(oldfile, 'r') as f:
        for line in f:
            if line[:6] == 'ENDMDL': # only take the first model
                break
            goodline = False
            for k in keepers:
                if line.startswith(k):
                    goodline=True
            if line[:6] in ['ATOM  ','HETATM'] and line[16] in ' A1': # altloc filter
                if line[17:20] in good_rsd_names:
                    goodline=True
                else:
                    if line[17:20] not in ['HOH']:
                        print( 'skip:',line[17:20],line[:-1])
                if goodline and line[17:20] not in standard_three_letter_codes:
                    print( 'WARNING: including funny resname:',line[17:20],'atom=',line[12:16])
            if line[:3] in ['TER','END']:
                goodline=True
            if goodline:
                if line.startswith('HETATM'):
                    print( 'WARNING: HETATM swap:',line[:-1])
                    out.write('ATOM  '+line[6:]) # switch HETATM to ATOM (eg MSE)
                else:
                    out.write(line)
    out.close()

def setup_pdb_protein_three_letter_codes():
    ''' Just called once to set up the tsv file
    '''

    outfile = Path.joinpath( path_to_db, 'pdb_codes.tsv')

    components_file = '/home/pbradley/csdat/pdb/components_2021-01-15.cif.gz'
    tmpfile1= components_file+'.tmp1'
    cmd = f"zgrep -F 'L-PEPTIDE LINKING' {components_file} -B5 -A20 > {tmpfile1}"
    print(cmd)
    system(cmd)


    tmpfile2= components_file+'.tmp2'
    cmd = f"egrep 'comp\.id|mon_nstd_parent|one_letter_code|three_letter_code' {tmpfile1} > {tmpfile2}"
    print(cmd)
    system(cmd)

    dfl = []
    for line in open(tmpfile2, 'r'):
        l = line.split()
        if l[0] == '_chem_comp.id':
            id = l[1]
            parent, code1 = None, None
        elif l[0] == '_chem_comp.mon_nstd_parent_comp_id':
            parent = l[1]
        elif l[0] == '_chem_comp.one_letter_code':
            code1 = l[1]
        elif l[0] == '_chem_comp.three_letter_code':
            code3 = l[1]
            assert parent and code1
            if parent in standard_three_letter_codes:
                dfl.append( dict(code3=code3, code1=code1, id=id, parent=parent))

    df = pd.DataFrame(dfl)
    df.to_csv(outfile, sep='\t', index=False)
    print('made:', outfile)

