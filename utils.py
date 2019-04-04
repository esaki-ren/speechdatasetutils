from scipy.signal import check_COLA


def make_stft_args(frame_period, fs, nperseg=None, window='hann', **kwargs):
    nshift = fs * frame_period // 1000
    if nperseg is None:
        nperseg = nshift * 2
    dct = dict(fs=fs, window=window, nperseg=nperseg, noverlap=nperseg - nshift)
    assert check_COLA(**dct)

    return dct
