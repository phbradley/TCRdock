import argparse

parser = argparse.ArgumentParser(
    description = "Read pdbfile(s) with MHC:peptide:TCRA:TCRB chains and parse some "
    "info, saving it in a TSV file",
    epilog = f'''The PDB files should have the following chains (in order):
    MHC-I:    (1) MHC, (2) beta-2 microglobulin, (3) peptide, (4) TCRA, (5) TCRB
    or MHC-I: (1) MHC, (2) peptide, (3) TCRA, (4) TCRB
    MHC-II:   (1) MHCA, (2) MHCB, (3) peptide, (4) TCRA, (5) TCRB
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument('--out_tsvfile', required=True, help='stuff')
parser.add_argument('--pdbfiles', nargs='*', help='stuff', required=True)
parser.add_argument('--mhc_class', type=int, help='stuff', required=True)
parser.add_argument('--organism', help='stuff', required=True)

args = parser.parse_args()

import pandas as pd
import tcrdock
import os

dfl = []

tmpfile = args.out_tsvfile+'.in_progress.tsv'

for fname in args.pdbfiles:
    print('start', fname, flush=True)
    pose = tcrdock.pdblite.pose_from_pdb(fname)
    num_chains = len(pose['chains'])
    if args.mhc_class==1:
        if num_chains == 5:
            # remove B2M
            print(f'removing chain 2 from 5-chain MHC class I pose')
            pose = tcrdock.pdblite.delete_chains(pose, [1]) # 0-indexed chain number
            num_chains = len(pose['chains'])
        else:
            assert num_chains==4, \
            f'MHC-I pdbfile {fname} should have 4 or 5 chains, see --help message'
        cs = pose['chainseq'].split('/')
        mhc_aseq, pep_seq, tcr_aseq, tcr_bseq = cs
        mhc_bseq = None
    else:
        assert num_chains==5, \
            f'MHC-II pdbfile {fname} should have 5 chains, see --help message'
        cs = pose['chainseq'].split('/')
        mhc_aseq, mhc_bseq, pep_seq, tcr_aseq, tcr_bseq = cs

    tdinfo = tcrdock.tcrdock_info.TCRdockInfo().from_sequences(
        args.organism, args.mhc_class, mhc_aseq, mhc_bseq, pep_seq, tcr_aseq, tcr_bseq)

    mhc_stub = tcrdock.mhc_util.get_mhc_stub(pose, tdinfo)
    tcr_stub = tcrdock.tcr_util.get_tcr_stub(pose, tdinfo)

    dgeom = tcrdock.docking_geometry.DockingGeometry().from_stubs(mhc_stub, tcr_stub)

    outl = {
        'pdbfile': fname,
        'organism': args.organism,
        'mhc_class': args.mhc_class,
        'sequence': pose['sequence'],
        'chainseq': pose['chainseq'],
        'tcrdock_info': tdinfo.to_string(),
        **dgeom.to_dict(),
        'tcr_frame_x_axis': ','.join(str(x) for x in tcr_stub['axes'][0]),
        'tcr_frame_y_axis': ','.join(str(x) for x in tcr_stub['axes'][1]),
        'tcr_frame_z_axis': ','.join(str(x) for x in tcr_stub['axes'][2]),
        'tcr_frame_origin': ','.join(str(x) for x in tcr_stub['origin']),
        'mhc_frame_x_axis': ','.join(str(x) for x in mhc_stub['axes'][0]),
        'mhc_frame_y_axis': ','.join(str(x) for x in mhc_stub['axes'][1]),
        'mhc_frame_z_axis': ','.join(str(x) for x in mhc_stub['axes'][2]),
        'mhc_frame_origin': ','.join(str(x) for x in mhc_stub['origin']),
    }
    dfl.append(outl)

    # show partial output
    pd.DataFrame(dfl).to_csv(tmpfile, sep='\t', index=False)

# make final output
pd.DataFrame(dfl).to_csv(args.out_tsvfile, sep='\t', index=False)
print('made:', args.out_tsvfile)

if os.path.exists(tmpfile):
    os.remove(tmpfile)


