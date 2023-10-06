from . import util
from . import mhc_util
from .tcrdock_info import TCRdockInfo
from . import pdblite


def load_and_setup_tcrdock_pose(
        pdb_filename,
        organism,
        mhc_class,
        class1_mhc_maxlen=181,
        class2_mhc_maxlen=95,
        class2_mhc_npad = 4,# residues before the first core position...
        pmhc_only = False,
):
    ''' Returns pose, TCRdockInfo

    class I:: 5 chains: MHC B2M peptide TCRa TCRb
    class I:: 4 chains: MHC peptide TCRa TCRb (already parsed)
    class II:: 5 chains: MHCa MHCb peptide TCRa TCRb

    returned pose will have 4 chains if class 1, 5 chains if class 2

    DOES renumber the pose/chains

    '''

    assert mhc_class in [1,2]

    pose = pdblite.pose_from_pdb(pdb_filename) # require_bb=True is the default

    if mhc_class == 1:
        assert ((not pmhc_only) and len(pose['chains']) in [4,5] or
                pmhc_only and len(pose['chains']) in [2,3])
        if pose['chainbounds'][1] > class1_mhc_maxlen:
            pose = pdblite.delete_residue_range(
                pose, class1_mhc_maxlen, pose['chainbounds'][1])
        if (not pmhc_only and len(pose['chains']) == 5 or
            pmhc_only and len(pose['chains']) == 3):
            # delete B2M
            pose = pdblite.delete_residue_range(
                pose, pose['chainbounds'][1], pose['chainbounds'][2])
    else:
        assert mhc_class == 2
        if (not pmhc_only and len(pose['chains']) != 5 or
            pmhc_only and len(pose['chains']) != 3):
            print('bad num chains for class 2 pose', pose['chains'],
                  pdb_filename)
            return None, None
        # trim the mhc chains, if necessary
        cs = pose['chainseq'].split('/')
        # core_posl is 0-indexed!
        core_posl = mhc_util.get_mhc_core_positions_class2(cs[0], cs[1])
        assert len(core_posl) == 12
        ntrim_alpha = core_posl[0] - class2_mhc_npad
        ntrim_beta = core_posl[6] - pose['chainbounds'][1] - class2_mhc_npad
        #print('load_and_setup_tcrdock_pose:: class2 ntrims:', ntrim_alpha, ntrim_beta)
        if ntrim_alpha>0:
            print('setup.load_and_setup_tcrdock_pose:: ntrim_alpha:', ntrim_alpha)
            pose = pdblite.delete_residue_range(pose, 0, ntrim_alpha)
        if ntrim_beta>0:
            print('setup.load_and_setup_tcrdock_pose:: ntrim_beta:', ntrim_beta)
            pose = pdblite.delete_residue_range(
                pose, pose['chainbounds'][1], pose['chainbounds'][1]+ntrim_beta)

        for ch in range(2):
            cb, ce = pose['chainbounds'][ch:ch+2]
            if ce-cb > class2_mhc_maxlen:
                print('setup.load_and_setup_tcrdock_pose: trim mhc class 2 chain:',
                      ch, ce-cb)
                pose = pdblite.delete_residue_range(
                    pose, cb+class2_mhc_maxlen, ce)

    assert len(pose['chains']) == 3+mhc_class-2*pmhc_only
    cs = pose['chainseq'].split('/')
    tcr_seqs = [None,None] if pmhc_only else cs[-2:]
    mhc_aseq = cs[0]
    mhc_bseq = '' if mhc_class==1 else cs[1]
    pep_seq = cs[mhc_class]
    tdi = TCRdockInfo().from_sequences(
        organism, mhc_class, mhc_aseq, mhc_bseq, pep_seq, *tcr_seqs)

    if not tdi.valid:
        print('load_and_setup_tcrdock_pose:: tdinfo parse error:', pdb_filename)
        return None, None

    if not pmhc_only:
        # now trim the ends of the tcr chains, if necessary
        for ich in range(2):
            ch = mhc_class+1+ich
            cb, ce = pose['chainbounds'][ch:ch+2]
            cdr_num = 3 + 4*ich # 3 or 7
            cdr3_stop = tdi.tcr_cdrs[cdr_num][1]
            #print(ch, cdr_num, tdi.tcr_cdrs[cdr_num], ch)
            maxpos = cdr3_stop + 3 + 8 # GXG then 8 rsds
            if ce > maxpos+1:
                print('setup.load_and_setup_tcrdock_pose: trimming tcr chain:',
                      ch, ce-(maxpos+1))
                pose = pdblite.delete_residue_range(pose, maxpos+1, ce)
                tdi.delete_residue_range(maxpos+1, ce)

    pose = pdblite.set_chainbounds_and_renumber(pose, list(pose['chainbounds']))

    pose = mhc_util.orient_pmhc_pose(pose, mhc_core_positions=tdi.mhc_core)

    return pose, tdi

