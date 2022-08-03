# import numpy as np
# import sys
# from os import system
import os.path
from os import system, popen
from os.path import exists
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

path_to_tcrdock = Path(__file__).parent
assert os.path.isdir( path_to_tcrdock )

path_to_db = path_to_tcrdock / 'db'
assert os.path.isdir( path_to_db )

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

long2short = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
              'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
              'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
              'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
              'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

standard_three_letter_codes = set(long2short.keys())


def torsion2abego(phi, psi, omega):
    if abs(omega) < 90:
        return 'O' # cis-omega
    elif phi >= 0.0:
        if -100 < psi and psi <= 100:
            return 'G' # alpha-L
        else:
            return 'E' # E
    else:
        if -125 < psi and psi <= 50:
            return 'A' # helical
        else:
            return 'B' # extended

def read_fields(filename):
    ''' Read a fields-formatted file where each line has repeated
    "key: value " texts
    returns a pandas dataframe
    '''
    data = open(filename,'r')
    line = data.readline()
    data.close()
    l = line.split()
    assert len(l)%2==0
    num_fields = len(l)//2
    fields = [l[2*i].replace(':','') for i in range(num_fields)]
    names = ['key'+str(i) if j==0 else fields[i]
             for i in range(num_fields)
             for j in range(2) ]
    usecols = [2*i+1 for i in range(num_fields)]
    return pd.read_csv(filename, sep='\s+', names = names, usecols=usecols)


# from
# https://stackoverflow.com/questions/47222585/matplotlib-generic-colormap-from-tab10
# still not quite right for gray, and the bright_first option doesnt work...
#
def categorical_cmap(nc, nsc, cmap="tab10", continuous=False, bright_first=True):
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        print(i,c,chsv)
        arhsv = np.tile(chsv,nsc).reshape(nsc,3)
        arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
        arhsv[:,2] = np.linspace(chsv[2],1,nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = rgb
    if bright_first:
        cols = cols.reshape(nc, nsc, 3).T.reshape(nc*nsc, 3)
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap
