import sys
from sys import exit
from os.path import exists, samefile
from os import system, getcwd
from pathlib import Path

## this is what should get created at the end
blast_foldername = 'ncbi-blast-2.11.0+' # this is what should get created


## here we make sure we are running the script in the correct place
cwd = Path(getcwd())
basedir = Path(__file__).parent
if not samefile(cwd, basedir):
    print('ERROR: Please run this script in the TCRdock/ directory')
    print('cwd:', cwd)
    print('TCRdock dir:', basedir)
    exit()

def download_web_file(address):
    newfile = address.split('/')[-1]

    if exists(newfile):
        print(f'download_web_file: {newfile} already exists, delete it to re-download')
        return

    ## try with wget
    cmd = 'wget '+address
    print(cmd)
    system(cmd)

    if not exists(newfile):
        print('wget failed, trying curl')
        cmd = 'curl -L {} -o {}'.format(address,newfile)
        print(cmd)
        system(cmd)

    if not exists(newfile):
        print('[ERROR] unable to download (tried wget and curl) the link', address)



if sys.platform == 'linux':
    address = ('https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.11.0/'
               'ncbi-blast-2.11.0+-x64-linux.tar.gz')
elif sys.platform == 'darwin':
    address = ('https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.11.0/'
               'ncbi-blast-2.11.0+-x64-macosx.tar.gz')
else:
    print('unrecognized platform type:', sys.platform,'expected "linux" or "darwin"')
    exit()


download_web_file(address)

tarfile = address.split('/')[-1]
assert exists(tarfile)

if not exists(blast_foldername):
    cmd = 'tar -xzf '+tarfile
    print(cmd)
    system(cmd)
else:
    print('folder already exists:', blast_foldername)


# right now these seem to be the executables we are using:
assert exists(Path(blast_foldername) / 'bin' / 'blastp')
assert exists(Path(blast_foldername) / 'bin' / 'makeblastdb')
