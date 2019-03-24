import os

from scipy import signal

FS = 16000
FRAME_PERIOD = 5  # [ms]
NSHIFT = FS * FRAME_PERIOD // 1000
NPERSEG = 960

DEFAULT = dict(
    window='hann',
    fs=FS,
    frame_period=FRAME_PERIOD,  # [ms]
    nshift=NSHIFT,
    nperseg=NPERSEG,
    noverlap=NPERSEG - NSHIFT,
    nmels=80,
    res_type='kaiser_best',
    minimum_len=8000,
    preemphasis_coef=None,  # preemphasis_coef=0.97,
    f_min=0,  # 70
    f_max=7600,
    rescaling=True,
    pad=0.2,  # [s]
)

if __name__ == '__main__':
    assert signal.check_COLA(
        DEFAULT['WINDOW'], DEFAULT['NPERSEG'], DEFAULT['NOVERLAP'])
