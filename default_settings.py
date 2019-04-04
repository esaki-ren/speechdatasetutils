
DEFAULT = dict(
    window='hann',
    fs=16000,
    frame_period=5,  # [ms]
    # nperseg=960,
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
    pass
