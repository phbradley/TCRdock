######################################################################################88
import argparse

required_columns = 'va cdr3a vb cdr3b'.split()

parser = argparse.ArgumentParser(
    description = "Calculate TCRdist distances between paired TCR sequences "
    "provided in an input TSV file (also compute TCRdiv repertoire diversity)",
    epilog = f'''

The TSV file should have the following columns
    va = TCR V-alpha gene including allele (e.g., "TRAV1-2*01")
    cdr3a = CDR3 alpha from 'C' to 'F/Y'
    vb = TCR V-beta gene including allele (e.g., "TRBV19*01")
    cdr3b = CDR3 beta from 'C' to 'F/Y'

Note: This python implementation is not super-fast. If you need to compute
    many TCRdist values, consider trying the tcrdist3 package or the C++
    implementation in the CoNGA github repository.

Example command line:

python compute_tcrdists.py --tcrs_tsvfile examples/tcrdist/human_tcrs.tsv \\
    --organism human --outfile tcrdists.txt
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument('--tcrs_tsvfile', required=True,
                    help='TSV-formatted input file with list of TCRs; see '
                    ' --help output for details')
parser.add_argument('--organism', required=True, choices=['mouse','human'],
                    help='TCR source organism')
parser.add_argument('--tcrdiv_sigma', type=float, default = 120.,
                    help='Width of gaussian smoothing kernel used in the TCRdiv '
                    'calculation (in TCRdist units)')
parser.add_argument('--outfile', required=True,
                    help='Output file where the distances will be written '
                    '(in numpy.savetxt format)')

args = parser.parse_args()

import numpy as np
import pandas as pd
import tcrdock

# load the docking geometry info
df = pd.read_table(args.tcrs_tsvfile)

missing = [col for col in required_columns if col not in df.columns]
if missing:
    print('ERROR missing some required columns in the --tcrs_tsvfile',
          missing)
    print('try running with the --help option for formatting info')
    exit()

print(f'Read {len(df)} TCRs from file {args.tcrs_tsvfile}')
# make a list of tcr tuples for providing to the tcrdist calculator
tcrs = [((l.va, '', l.cdr3a), (l.vb, '', l.cdr3b)) # we can leave J gene empty 4 TCRdist
        for l in df.itertuples()]

# get the TCRdist calculator
tcrdister = tcrdock.sequtil.get_tcrdister(args.organism)

N = len(tcrs)
D = np.array([tcrdister(x,y) for x in tcrs for y in tcrs]).reshape((N,N))

np.savetxt(args.outfile, D, fmt='%.1f')
print(f'saved distance matrix to {args.outfile}')

# now compute TCRdiv
D[np.arange(N), np.arange(N)] = 1e6 # set diagonal to a very large value

sdev = args.tcrdiv_sigma
tcrdiv = -1*np.log(np.sum(np.exp(-1*(D/sdev)**2))/(N*(N-1)))
print(f'TCRdiv for input TCR repertoire: {tcrdiv:.3f}')


