from scipy.signal import check_COLA


def make_stft_args(frame_period, nperseg, fs, window='hann', **kwargs):
    nshift = fs * frame_period // 1000
    dct = dict(window='hann', nperseg=256, noverlap=nperseg - nshift)
    assert check_COLA(**dct)

    return dct
