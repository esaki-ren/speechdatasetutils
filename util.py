from scipy.signal import check_COLA


def make_stft_args(frame_period, fs, nperseg=None, window='hann', **kwargs):
    nshift = fs * frame_period // 1000
    if nperseg is None:
        nperseg = nshift * 2
    dct = dict(window='hann', nperseg=256, noverlap=nperseg - nshift)
    assert check_COLA(**dct)

    return dct
