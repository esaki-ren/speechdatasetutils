import os

from scipy import signal

DATASETROOT = os.path.expanduser(os.path.join('~', 'dataset'))

FS = 16000
FRAME_PERIOD = 5  # [ms]
NSHIFT = FS * FRAME_PERIOD // 1000
NPERSEG = 960

DEFAULT = dict(
    datasetroot = DATASETROOT,
    wavdir = 'wav',
    npzdir = 'npz',

    window = 'hann',
    fs = FS,
    frame_period = FRAME_PERIOD,  # [ms]
    nshift = NSHIFT,
    nperseg = NPERSEG,
    noverlap = NPERSEG - NSHIFT,
    nmels = 80,
    res_type='scipy',
    minimum_len=10000,
    preemphasis=0.97,
)
if __name__ == '__main__':
    assert signal.check_COLA(DEFAULT['WINDOW'], DEFAULT['NPERSEG'], DEFAULT['NOVERLAP'])
