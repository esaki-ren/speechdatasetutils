from scipy.signal import check_COLA


def make_stft_args(frame_period, fs, nperseg=None, window='hann', **kwargs):
    nshift = fs * frame_period // 1000

    if nperseg is None:
        nperseg = nshift * 2

    noverlap = nperseg - nshift

    dct = dict(window=window, nperseg=nperseg, noverlap=noverlap)
    assert check_COLA(**dct)

    dct["fs"] = fs
    return dct
