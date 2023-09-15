# rebuild 1000s of random loops into BOTH CDRs at the same time
# allow variable length CDRs
# vary the sequence between CAX and XF
# provide template info for CAXX and XXF
#


import argparse
parser = argparse.ArgumentParser(description="alphafold loop design")

parser.add_argument('--num_runs', type=int, required=True)
parser.add_argument('--outfile', required=True)
parser.add_argument('--template_pdbids', nargs='*', required=True)
parser.add_argument('--peptides', nargs='*')
parser.add_argument('--cdr3a_lendiff_bias', type=float, nargs='*',
                    default= [1.0])
parser.add_argument('--cdr3b_lendiff_bias', type=float, nargs='*',
                    default= [1.0])
parser.add_argument('--make_batches', action='store_true')
parser.add_argument('--rundir')
parser.add_argument('--batch_size', type=int, default=25)
parser.add_argument('--loop_design_extra_args', default = '')

args = parser.parse_args()

if args.make_batches:
    assert args.rundir

lendiff_bias = {}
for ab, bias in zip('AB',[args.cdr3a_lendiff_bias, args.cdr3b_lendiff_bias]):
    assert len(bias)%2==1
    lendiff_bias[ab] = bias

# other imports
import design_paths
design_paths.setup_import_paths()
import tcrdock as td2
import pandas as pd
from os.path import exists
from os import mkdir
import random
from collections import Counter

## hard-coded
match_seq_stems = False
nterm_seq_stem = 3
cterm_seq_stem = 2
nterm_align_stem = 4
cterm_align_stem = 3
force_native_seq_stems = True # since these aren't being designed...
## defaults ##########

def cdr3s_match(cdr3_0, cdr3_1):
    ''' I.e. they are close enough to exchange one for the other during modeling
    '''
    global match_seq_stems, nterm_seq_stem, cterm_seq_stem
    #lendiff = abs(len(cdr3_0) - len(cdr3_1))
    return (not match_seq_stems or
            (cdr3_0[:nterm_seq_stem]  == cdr3_1[:nterm_seq_stem] and
             cdr3_0[-cterm_seq_stem:] == cdr3_1[-cterm_seq_stem:]))

tcrs = td2.sequtil.ternary_info[
    td2.sequtil.ternary_info.pdbid.isin(args.template_pdbids)].copy()
assert tcrs.shape[0] == len(args.template_pdbids)

print('num pdb targets:', tcrs.shape[0])

tcrs_file = design_paths.PAIRED_TCR_DB
print('reading:', tcrs_file)
big_tcrs_df = pd.read_table(tcrs_file)

targets_dfl = []

for l in tcrs.itertuples():
    template_pdbfile = str(td2.util.path_to_db / l.pdbfile)
    pose = td2.pdblite.pose_from_pdb(template_pdbfile)
    # confirm pdb numbering is 0-indexed sequence numbers
    assert all((int(r[1])==i for i,r in enumerate(pose['resids'])))
    tdifile = template_pdbfile+'.tcrdock_info.json'
    with open(tdifile, 'r') as f:
        tdinfo = td2.tcrdock_info.TCRdockInfo().from_string(f.read())

    nat_sequence = pose['sequence']
    nat_chainseq = pose['chainseq']
    nat_peptide = nat_chainseq.split('/')[1]
    assert nat_chainseq.replace('/', '') == nat_sequence

    #cbs = pose['chainbounds']
    nres = len(nat_sequence)
    #a,b,c = cbs[:3]

    cdr3_bounds = [tdinfo.tcr_cdrs[3], tdinfo.tcr_cdrs[7]]
    cdr3a, cdr3b = [nat_sequence[start:stop+1] for start,stop in cdr3_bounds]
    assert nat_sequence.count(cdr3a) == 1 and nat_sequence.count(cdr3b) == 1

    all_random_cdr3s = []
    for ab, cdr3 in zip('AB', [cdr3a, cdr3b]):
        bias = lendiff_bias[ab]
        assert len(bias)%2==1
        shift = len(bias)//2
        bias = dict( (i-shift,b) for i,b in enumerate(bias))
        vals, weights = zip(*bias.items())
        lendiffs = random.choices(vals, weights=weights, k=args.num_runs)
        random_cdr3s = []
        for lendiff, count in Counter(lendiffs).most_common():
            possibles = [x for x in big_tcrs_df['cdr3'+ab.lower()]
                         if len(x)-len(cdr3) == lendiff and
                         cdr3s_match(x, cdr3)]
            print('cdr3'+ab, 'lendiff=', lendiff, 'want:', count,
                  'have:', len(possibles))
            random_cdr3s.extend(random.choices(possibles, k=count))#allow replacmnt
        assert len(random_cdr3s) == args.num_runs
        random.shuffle(random_cdr3s) # doh! forgot to do this for altest18/run281!
        if force_native_seq_stems:
            random_cdr3s = [
                cdr3[:nterm_seq_stem] + x[nterm_seq_stem:-cterm_seq_stem] +
                cdr3[-cterm_seq_stem:] for x in random_cdr3s]
        all_random_cdr3s.append(random_cdr3s)

    base_alignment = list(range(nres)) # the numbers in here are template posns
    for start, stop in cdr3_bounds:
        #unalign the whole cdr3 MINUS the nterm_ and cterm_ align_stems
        for pos in range(start+nterm_align_stem, stop+1-cterm_align_stem):
            base_alignment[pos] = -1
    assert base_alignment.count(-1) == (
        len(cdr3a) + len(cdr3b) - 2*nterm_align_stem - 2*cterm_align_stem)

    if args.peptides:
        peptides = args.peptides[:]
    else:
        peptides = [nat_peptide]

    for peptide in peptides:
        for ii, (new_cdr3a, new_cdr3b) in enumerate(zip(*all_random_cdr3s)):
            chainseq = nat_chainseq.replace(cdr3a,new_cdr3a)\
                                   .replace(cdr3b,new_cdr3b)\
                                   .replace(nat_peptide,peptide)
            sequence = chainseq.replace('/','')

            alignment = base_alignment[:]
            shifts = [len(new_cdr3a) - len(cdr3a), len(new_cdr3b) - len(cdr3b)]

            offset = 0
            for shift, bounds in zip(shifts, cdr3_bounds):
                m1_start = bounds[0] + nterm_align_stem + offset # start of minus 1s
                assert alignment[m1_start]==-1 and alignment[m1_start-1]!=-1
                if shift>0:
                    alignment = alignment[:m1_start]+[-1]*shift+alignment[m1_start:]
                elif shift<0:
                    alignment = alignment[:m1_start] + alignment[m1_start-shift:]
                offset += shift

            assert len(alignment) == len(sequence)
            assert alignment.count(-1) == (
                len(new_cdr3a)+len(new_cdr3b)-2*nterm_align_stem-2*cterm_align_stem)

            alignstring = ';'.join(f'{i}:{j}' for i,j in enumerate(alignment)
                                   if j != -1)

            outl = dict(
                targetid = f'{l.pdbid}_tcr_{ii}_{peptide}',
                chainseq = chainseq,
                template_0_template_pdbfile = template_pdbfile,
                template_0_target_to_template_alignstring = alignstring,
                cdr3a = new_cdr3a,
                wt_cdr3a = cdr3a,
                ashift = shifts[0],
                cdr3b = new_cdr3b,
                wt_cdr3b = cdr3b,
                bshift = shifts[1],
                peptide = peptide,
                pdbid = l.pdbid,
            )
            targets_dfl.append(outl)

targets = pd.DataFrame(targets_dfl)

targets.to_csv(args.outfile, sep='\t', index=False)
print('made:', args.outfile)

if args.make_batches:
    import pandas as pd
    from os.path import exists
    from pathlib import Path

    PY = design_paths.LOOP_DESIGN_PYTHON
    EXE = (Path(__file__).parent / 'loop_design.py').resolve()
    assert exists(EXE) and exists(PY)

    rundir = args.rundir
    if not exists(rundir):
        mkdir(rundir)

    rundir = str(Path(rundir).resolve()) # want absolute paths in cmds_file
    if not rundir.endswith('/'):
        rundir += '/'


    cmds_file = f'{rundir}commands.txt'
    assert not exists(cmds_file)

    out = open(cmds_file, 'w')

    targets['nres'] = (targets.chainseq.str.len()-
                       targets.chainseq.str.count('/'))
    targets.sort_values('nres', inplace=True)

    num_batches = (targets.shape[0]-1)//args.batch_size + 1

    for b in range(num_batches):
        outfile = f'{rundir}batch_{b}_targets.tsv'
        targets.iloc[args.batch_size*b:args.batch_size*(b+1)].to_csv(
            outfile, sep='\t', index=False)
        outfile_prefix = f'{rundir}batch_{b}'
        cmd = (f'{PY} {EXE} {args.loop_design_extra_args} --targets {outfile} '
               f' --outfile_prefix {outfile_prefix} '
               f' > {outfile_prefix}.log 2> {outfile_prefix}.err')
        out.write(cmd+'\n')
        if b%100==0:
            print('made batch targets file:', b, outfile)
    out.close()
    print('made:', cmds_file)
    exit()
