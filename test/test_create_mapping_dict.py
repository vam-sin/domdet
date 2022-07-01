from get_pae import create_mapping_dict, get_alignment

def test_create_mapping_dict1():
    s1 = 'ABCDEFGH'
    alignment = {
        'seqA':s1,
        'seqB':s1
    }
    mapping = create_mapping_dict(alignment)
    assert set(mapping.keys()) == set(mapping.values())
    assert set(mapping.keys()) == set(range(len(s1)))
    assert all([k==v for k,v in mapping.items()])

def test_create_mapping_dict2():
    pdb_seq =  '-B-DE'
    up_seq =   'ABCDE'
    alignment = {
        'seqA':up_seq,
        'seqB':pdb_seq
    }
    mapping = create_mapping_dict(alignment)
    assert mapping == {0:1, 1:3, 2:4}

def test_create_mapping_dict3():
    pdb =     'XYZQTP'
    unip = 'ABCXYZQQPABC'
    align = get_alignment(unip, pdb)
    mapping = create_mapping_dict(align)
    assert mapping == {0:3, 1:4, 2:5, 3:6, 4:7, 5:8}

def test_create_mapping_dict4():
    pdb =     'XYZQQQ'
    unip = 'ABCXYZAAAQQQABC'
    align = get_alignment(unip, pdb)
    mapping = create_mapping_dict(align)
    assert mapping == {0:3, 1:4, 2:5, 3:9, 4:10, 5:11}


def test_get_alignment1():
    up_seq =  'ZZABCZZ'
    pdb_seq =   'ABC'
    align = get_alignment(up_seq, pdb_seq)
    assert align['seqA'] == 'ZZABCZZ'
    assert align['seqB'] == '--ABC--'

def test_get_alignment2():
    up_seq =  'ZZABCZZ'
    pdb_seq =   'ATC'
    align = get_alignment(up_seq, pdb_seq)
    assert align['seqA'] == 'ZZABCZZ'
    assert align['seqB'] == '--ATC--'

def test_get_alignment3():
    up_seq =  'ZZABCZZ'
    pdb_seq = 'XXATCXX'
    align = get_alignment(up_seq, pdb_seq)
    assert align['seqA'] == 'ZZABCZZ'
    assert align['seqB'] == 'XXATCXX'

def test_get_alignment4():
    up_seq =  'PABCD'
    pdb_seq = 'P'
    align = get_alignment(up_seq, pdb_seq)
    assert align['seqA'] == 'PABCD'
    assert align['seqB'] == 'P----'

def test_get_alignment5():
    up_seq =  'PABCD'
    pdb_seq = 'X'
    align = get_alignment(up_seq, pdb_seq)
    assert align['seqA'] == 'PABCD'
    assert align['seqB'] == '----X'

def test_get_alignment6():
    up_seq =  'PABC'
    pdb_seq = 'ABCD'
    align = get_alignment(up_seq, pdb_seq)
    assert align['seqA'] == 'PABC-'
    assert align['seqB'] == '-ABCD'

def test_get_alignment7():
    up_seq =  'NOPEHEADDRINKOTHER'
    pdb_seq =         'DRUNK'
    align = get_alignment(up_seq, pdb_seq)
    assert align['seqA'] == 'NOPEHEADDRINKOTHER'
    assert align['seqB'] == '--------DRUNK-----'

# def test_create_mapping_dict3():
#     pdb_seq =  '_B_DEFG'
#     up_seq =   'ABCDE_G'
#     alignment = {
#         'seqA':up_seq,
#         'seqB':pdb_seq
#     }
#     mapping = create_mapping_dict(alignment)
#     assert mapping == {0:1, 1:3, 2:4, 3:}
