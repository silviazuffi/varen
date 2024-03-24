'''
    List of the frames for which we have prototypes
'''

def horse_capture_date(horse):
    if horse in ['H0001','H0002','H0003','H0004','H0005','H0006','H0007']:
        scans_d='20221006' 
    if horse in ['H0008','H0009','H0010']:
        scans_d='20221013'
    if horse in ['H0011','H0012','H0013','H0014']:
        scans_d='20221014'
    if horse in ['H0015','H0016','H0017','H0018','H0019','H0020','H0021','H0022' ]:
        scans_d='20221020'
    if horse in ['H0023','H0024']:
        scans_d='20221021'
    if horse in ['H0025','H0026','H0027','H0028']:
        scans_d='20221026'
    if horse in ['H0029','H0030','H0031']:
        scans_d='20221027'
    if horse in ['H0032','H0033']:
        scans_d='20221028'
    if horse in ['H0034','H0035','H0036','H0037']:
        scans_d='20221103'
    return scans_d


prototype_clip = {
    'H0001':'still1',
    'H0002':'still1',
    'H0004':'still1',
    'H0005':'still1',
    'H0006':'still1',
    'H0007':'still1',
    'H0008':'still1',
    'H0009':'still1',
    'H0011':'still1',
    'H0013':'still1',
    'H0015':'head1',
    'H0016':'leg1',
    'H0017':'head1',
    'H0017':'still1',
    'H0020':'head1',
    'H0021':'head1',
    'H0022':'still1',
    'H0023':'head1',
    'H0027':'still1',
    'H0028':'still1',
    'H0030':'still1',
    'H0031':'head1',
    'H0034':'still2',
    'H0035':'still2',
    'H0037':'still1_018_164',
    }
'''
    'H0038':'still1',
    'H0039':'still1',
    'H0040':'still1',
    'H0041':'still1',
    'H0042':'still1',
    'H0043':'still1',
    'H0044':'still1',
    'H0045':'head1',
    'H0046':'still1',
    'H0047':'still1',
    'H0052':'still1',
    'H0053':'still1',
    'H0054':'other1',
    'H0055':'still1',
}
'''
prototype_frames = {
    'H0001_still1': [8],
    'H0002_still1': [2],
    'H0004_still1': [4],
    'H0005_still1': [4],
    'H0006_still1': [20],
    'H0007_still1': [13],
    'H0008_still1': [1],
    'H0009_still1': [1],
    'H0011_still1': [96],
    'H0013_still1': [19],
    'H0015_head1':  [45],
    'H0016_leg1':   [1],
    'H0017_head1':  [30],
    'H0017_still1': [92],
    'H0020_head1':  [187],
    'H0021_head1':  [119],
    'H0022_still1': [62],
    'H0023_head1':  [294],
    'H0027_still1': [279],
    'H0028_still1': [186],
    'H0030_still1': [5],
    'H0031_head1':  [285],
    'H0034_still2': [75],
    'H0035_still2': [35],
    'H0037_still1_018_164': [7],
    'H0038_still1': [10],
    'H0039_still1': [100],
    'H0040_still1': [1],
    'H0041_still1': [27],
    'H0042_still1': [36],
    'H0043_still1': [75],
    'H0044_still1': [147],
    'H0045_head1':  [182],
    'H0046_still1': [161],
    'H0047_still1': [113],
    'H0052_still1': [32],
    'H0053_still1': [105],
    'H0054_other1': [52],
    'H0055_still1': [12],
}

