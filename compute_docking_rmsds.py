import argparse

required_columns = 'torsion d tcr_unit_y tcr_unit_z mhc_unit_y mhc_unit_z'.split()

parser = argparse.ArgumentParser(
    description = "Calculate docking RMSDs from a TSV file containing docking "
    "geometries, for example one generated by the parse_tcr_pmhc_pdbfile.py script",
    epilog = f'''
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument('--docking_geometries_tsvfile', required=True, help='stuff')
parser.add_argument('--outfile', help='stuff', required=True)

args = parser.parse_args()

import numpy as np
import pandas as pd
from tcrdock.docking_geometry import (
    DockingGeometry, compute_docking_geometries_distance_matrix,
)

# load the docking geometry info
df = pd.read_table(args.docking_geometries_tsvfile)

missing = [col for col in required_columns if col not in df.columns]
if missing:
    print('ERROR missing some required columns in the --docking_geometries_tsvfile',
          missing)
    exit()

dgeoms = [DockingGeometry().from_dict(x) for _,x in df.iterrows()]

D = compute_docking_geometries_distance_matrix(dgeoms, dgeoms)

np.savetxt(args.outfile, D, fmt='%.6f')
print(f'saved distance matrix to {args.outfile}')
