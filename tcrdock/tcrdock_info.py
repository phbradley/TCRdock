################################################################################
from . import tcr_util
from . import mhc_util
from . import tcrdist
import json

class TCRdockInfo():
    ''' A class to store information about a TCR:peptide:MHC pose

    CONTENTS:

    * organism
    * mhc_class
    * mhc_allele
    * pep_seq
    * tcr = ((va,ja,cdr3a),(vb,jb,cdr3b))
    * mhc_core: mhc core positions
    * tcr_core: tcr core positions
    * tcr_cdrs: list of 8 [start,stop] (inclusive) 0-indexed position lists

    all positions are 0-indexed wrt full merged pose

    note that the CDR3 sequence and loop position is "extended"
    ie starts at C and ends at "F"
    '''

    def __init__(self):
        self.organism = None
        self.mhc_class = None
        self.mhc_allele = None
        self.pep_seq = None
        self.tcr = None
        self.mhc_core = None
        self.tcr_core = None
        self.tcr_cdrs = None
        self.valid = False

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

    def from_sequences(
            self,
            organism,
            mhc_class, # pass None if tcr_only
            mhc_aseq, # '' if tcr_only
            mhc_bseq, # ignored if mhc_class==1
            pep_seq, # '' if tcr_only
            tcr_aseq,
            tcr_bseq,
    ):
        self.valid = False
        self.organism = organism
        self.mhc_class = mhc_class if mhc_class is None else int(mhc_class)
        self.pep_seq = pep_seq
        #self.mhc_aseq = mhc_aseq
        #self.mhc_bseq = mhc_bseq
        #self.tcr_aseq = tcr_aseq
        #self.tcr_bseq = tcr_bseq

        # parse the mhc sequence
        if mhc_class is None: # tcr only
            assert mhc_aseq == '' and mhc_bseq == '' and pep_seq == ''
            self.mhc_core = []
            self.mhc_allele = ''

        elif mhc_class == 1:
            self.mhc_core = mhc_util.get_mhc_core_positions_class1(mhc_aseq)
            self.mhc_allele = mhc_util.get_mhc_allele(mhc_aseq, organism)

        else:
            assert mhc_class == 2
            self.mhc_core = mhc_util.get_mhc_core_positions_class2(
                mhc_aseq, mhc_bseq)
            self.mhc_allele = (mhc_util.get_mhc_allele(mhc_aseq, organism)+','+
                               mhc_util.get_mhc_allele(mhc_bseq, organism))

        if -1 in self.mhc_core:
            print('TCRdockInfo:: MHC parse fail!')
            return self ########## early return

        # this shifts everything to full pose numbering (0-indexed)
        offset = len(mhc_aseq) + len(pep_seq)
        if mhc_class==2:
            offset += len(mhc_bseq)

        self.tcr_cdrs = []
        self.tcr_core = []
        self.tcr = []
        for chain, chainseq in zip('AB', [tcr_aseq, tcr_bseq]):
            res = tcrdist.parsing.parse_tcr_sequence(
                organism, chain, chainseq)
            if not res:
                # parse failure
                print('TCRdockInfo:: TCR parse fail!')
                return self ########## early return
            cdr_loops = res['cdr_loops']
            cdr3_start, cdr3_end = cdr_loops[-1]
            cdr3 = chainseq[cdr3_start:cdr3_end+1]
            self.tcr.append((res['v_gene'], res['j_gene'], cdr3))
            # the ints here are converting from np.int64 to int
            # ran into trouble serializing:
            #  "Object of type 'int64' is not JSON serializable"
            self.tcr_cdrs.extend([(int(x[0]+offset), int(x[1]+offset))
                                  for x in cdr_loops])
            self.tcr_core.extend([int(x+offset) for x in res['core_positions']])

            offset += len(chainseq)
        self.valid = True # signal success
        return self

    def renumber(self, old2new):
        ''' old2new is a mapping of 0-indexed positions from old to new
        numbering systems

        will be bad if old2new doesn't cover all our positions
        '''
        old2new = {int(x):int(y) for x,y in old2new.items()} # no int64 in our data!
        if self.mhc_core is not None:
            self.mhc_core = [old2new[x] for x in self.mhc_core]
        if self.tcr_core is not None:
            self.tcr_core = [old2new[x] for x in self.tcr_core]
        if self.tcr_cdrs is not None:
            self.tcr_cdrs = [(old2new[x], old2new[y]) for x,y in self.tcr_cdrs]
        return self

    def delete_residue_range(self, start, stop):
        ''' delete from start to stop , EXCLUSIVE of stop !!!!!!!!!!!!!!!!!!!!!!
        ie delete [start,stop) or range(start,stop)

        puts -1 for residues in [start,stop) range, if there are any
        '''

        maxpos = self.tcr_cdrs[-1][-1] # end of CDR3beta is biggest
        numdel = stop-start
        old2new = {x:x if x<start else -1 if x<stop else x-numdel
                   for x in range(maxpos+1)}
        self.renumber(old2new)
        return self


    def to_dict(self):
        ''' should be inverse of from_dict
        '''
        return dict(
            organism = self.organism,
            mhc_class = self.mhc_class,
            mhc_allele = self.mhc_allele,
            mhc_core = self.mhc_core,
            pep_seq = self.pep_seq,
            tcr = self.tcr,
            tcr_core = self.tcr_core,
            tcr_cdrs = self.tcr_cdrs,
            valid = self.valid,
        )

    def from_dict(self, D):
        ''' should be inverse of to_dict
        '''
        self.organism = D['organism']
        try:
            self.mhc_class = int(D['mhc_class'])
        except TypeError:
            self.mhc_class = D['mhc_class']
        self.mhc_allele = D['mhc_allele']
        self.mhc_core = D['mhc_core']
        self.pep_seq = D['pep_seq']
        self.tcr = D['tcr']
        self.tcr_core = D['tcr_core']
        self.tcr_cdrs = D['tcr_cdrs']
        self.valid = D.get('valid', True)
        return self

    def to_string(self):
        return json.dumps(self.to_dict(), separators=(',', ':'))

    def from_string(self, info):
        return self.from_dict(json.loads(info))

