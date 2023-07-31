######################################################################################88

from collections import Counter
import pandas as pd
import numpy as np
import design_paths
design_paths.setup_import_paths()
import tcrdock as td2
from os import system, mkdir
from os.path import exists
import itertools as it
import json
import random

from design_stats import get_designable_positions

def setup_for_protein_mpnn(
        pose,
        design_mask,
        idstring, # like a pdbid, for example
):
    ''' returns 3 dicts: parsed_chains, assigned_chains, fixed_positions
    '''

    # which chains are designable?
    #
    # assigned_chains looks like:
    #{"4YOW": [["A", "C"], ["B", "D", "E", "F"]], "3HTN": [["A", "C"], ["B"]]}
    #
    # fixed_positions looks like: (these are 1-indexed)
    #{"4YOW": {"A": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,...
    #
    nres = len(pose['resids'])
    assert len(design_mask) == nres # list or array of True,False
    design_chains = sorted(set(r[0] for r,m in zip(pose['resids'], design_mask) if m))
    fixed_chains = sorted(set(c for c in pose['chains'] if c not in design_chains))

    assigned_chains = {idstring: [design_chains, fixed_chains]}

    fixed = {x:[] for x in pose['chains']}
    res_count = Counter()
    for r, m in zip(pose['resids'], design_mask):
        ch = r[0]
        res_count[ch] += 1
        if not m: # fixed
            fixed[ch].append(res_count[ch]) # 1-indexed!

    fixed_positions = {idstring:fixed}

    parsed_chains = {}
    cs = pose['chainseq'].split('/')
    atoms = [' N  ', ' CA ', ' C  ', ' O  ']

    for chain, seq in zip(pose['chains'], cs):
        parsed_chains['seq_chain_'+chain] = seq
        coords = {}
        for atom in atoms:
            xyzs = []
            nans = np.full(3,np.nan)
            for r in pose['resids']:
                if r[0] == chain:
                    xyzs.append(pose['coords'][r].get(atom, nans).tolist())
            coords[f'{atom.strip()}_chain_{chain}'] = xyzs
        parsed_chains['coords_chain_'+chain] = coords
    parsed_chains['name'] = idstring
    parsed_chains['num_of_chains'] = len(pose['chains'])
    parsed_chains['seq'] = ''.join(cs)


    return parsed_chains, assigned_chains, fixed_positions

def run_alphafold(
        targets,
        outprefix,
        num_recycle = 1,
        model_name = 'model_2_ptm', # want PAEs
        model_params_file = None,
):
    ''' Returns results df
    results are generated by predict_tcr_from_template_alignments.py, the
    *_final.tsv file  WITH EXTRA COLUMNS:

    model_pdbfile
    model_plddtfile
    model_paefile
    '''
    PY = design_paths.AF2_PYTHON
    EXE = design_paths.path_to_design_scripts / 'af2_wrapper.py'

    outfile = outprefix+'_targets.tsv'
    targets.to_csv(outfile, sep='\t', index=False)

    xargs = (f' --num_recycle {num_recycle} --model_names {model_name} ')
    if model_params_file is not None:
        xargs += f' --model_params_files {model_params_file} '

    cmd = (f'{PY} {EXE} {xargs} --targets {outfile} --outfile_prefix {outprefix} '
           f' > {outprefix}_run.log 2> {outprefix}_run.err')

    print(cmd, flush=True)
    system(cmd)

    resultsfile = outprefix+'_final.tsv'
    assert exists(resultsfile), 'run_alphafold failed '+outprefix

    dfl = []
    for _,l in pd.read_table(resultsfile).iterrows():
        outl = l.copy()
        pdbfile = l[model_name+'_pdb_file']
        plddtfile = l[model_name+'_plddt_file']
        paefile   = l[model_name+'_predicted_aligned_error_file']
        assert exists(pdbfile) and exists(paefile)
        outl['model_pdbfile'] = pdbfile
        outl['model_plddtfile'] = plddtfile
        outl['model_paefile'] = paefile
        dfl.append(outl)

    return pd.DataFrame(dfl)


def run_mpnn(
        targets,
        outprefix,
        num_mpnn_seqs=3,
        extend_flex=1,
):
    ''' Returns results df

    targets has to have the columns below

    '''
    required_cols = 'targetid chainseq model_pdbfile'.split()
    for col in required_cols:
        assert col in targets.columns

    # need one of these to figure out which positions are designable
    assert ('template_0_target_to_template_alignstring' in targets.columns or
            'designable_positions' in targets.columns)

    outdir = outprefix+'_mpnn/'
    if not exists(outdir):
        mkdir(outdir)

    parsed_chains, assigned_chains, fixed_positions = [], {}, {}

    for l in targets.itertuples():
        nres = len(l.chainseq.replace('/',''))
        flex_posl = get_designable_positions(row=l, extend_flex=extend_flex)
        design_mask = np.full((nres,), False)
        design_mask[flex_posl] = True

        cs = l.chainseq.split('/')
        chainbounds = [0] + list(it.accumulate(len(x) for x in cs))
        pose = td2.pdblite.pose_from_pdb(l.model_pdbfile)
        assert pose['sequence'] == ''.join(cs)
        pose = td2.pdblite.set_chainbounds_and_renumber(pose, chainbounds)
        assert len(pose['sequence']) == nres

        pc, ac, fp = setup_for_protein_mpnn(pose, design_mask, l.targetid)
        parsed_chains.append(pc)
        assigned_chains.update(ac)
        fixed_positions.update(fp)

    assigned_chains = [assigned_chains]
    fixed_positions = [fixed_positions]

    for vals, tag in zip([parsed_chains, assigned_chains, fixed_positions],
                         ['pc','ac','fp']):
        outfile = f'{outprefix}_{tag}.jsonl'
        with open(outfile,'w') as f:
            for val in vals:
                f.write(json.dumps(val) + '\n')

    cmd = (f'{design_paths.MPNN_PYTHON} {design_paths.MPNN_SCRIPT} '
           f' --jsonl_path            {outprefix}_pc.jsonl '
           f' --chain_id_jsonl        {outprefix}_ac.jsonl '
           f' --fixed_positions_jsonl {outprefix}_fp.jsonl '
           f' --out_folder {outdir} --num_seq_per_target {num_mpnn_seqs} '
           f' --sampling_temp "0.1" '
           f' --seed 37 --batch_size 1 > {outprefix}_run.log '
           f' 2> {outprefix}_run.err')
    print(cmd, flush=True)
    system(cmd)

    # now setup new targets array with redesigned sequences
    dfl = []
    for _, l in targets.iterrows():
        nres = len(l.chainseq.replace('/',''))
        flex_posl = get_designable_positions(row=l, extend_flex=extend_flex)
        design_mask = np.full((nres,), False)
        design_mask[flex_posl] = True
        cs = l.chainseq.split('/')
        old_fullseq = ''.join(cs)
        chainbounds = [0] + list(it.accumulate(len(x) for x in cs))

        fastafile = f'{outdir}/seqs/{l.targetid}.fa'
        with open(fastafile,'r') as f:
            fasta = f.read()
        seqs = [x for x in fasta.split('\n') if x and x[0] != '>'][1:]
        assert len(seqs) == num_mpnn_seqs
        seq_counts = Counter(seqs)
        _, top_count = seq_counts.most_common(1)[0]
        top_seq = random.choice([x for x,y in seq_counts.most_common()
                                 if y==top_count])
        top_seqs = top_seq.split('/')
        def seq_match_score(seq1, seq2):
            return 100*abs(len(seq1)-len(seq2)) + sum(x!=y for x,y in zip(seq1,seq2))
        for top_seq in top_seqs:
            ch = min(enumerate(cs), key=lambda x:seq_match_score(x[1],top_seq))[0]
            old_seq = cs[ch]
            assert len(old_seq) == len(top_seq)
            start = chainbounds[ch]
            assert all(((a==b) or c)
                       for a,b,c in zip(old_seq, top_seq, design_mask[start:]))
            cs[ch] = top_seq

        new_fullseq = ''.join(cs)
        outl = l.copy()
        outl['chainseq'] = '/'.join(cs) # update with designed sequence

        ## update cdr3 information in the output row
        if hasattr(l, 'cdr3a'):
            assert old_fullseq.count(l.cdr3a) == 1
            start = old_fullseq.index(l.cdr3a)
            new_cdr3a = new_fullseq[start:start+len(l.cdr3a)]
            outl['cdr3a'] = new_cdr3a

            assert old_fullseq(l.cdr3b)
            start = old_fullseq.index(l.cdr3b)
            new_cdr3b = new_fullseq[start:start+len(l.cdr3b)]
            outl['cdr3b'] = new_cdr3b

        dfl.append(outl)

    targets = pd.DataFrame(dfl)
    return targets # same as starting targets except for chainseq column

