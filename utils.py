from scipy.signal import check_COLA
from .preprocess import normalize


def make_stft_args(frame_period, fs, nperseg=None, window='hann', **kwargs):
    nshift = fs * frame_period // 1000

    if nperseg is None:
        nperseg = nshift * 4

    noverlap = nperseg - nshift

    dct = dict(window=window, nperseg=nperseg, noverlap=noverlap)
    if not check_COLA(**dct):
        raise ValueError(dct)

    dct["fs"] = fs
    return dct


def chech_gain(wave, fs, gain, cutoff):

    wave = normalize(
        wave, fs, dc_removal=True,
        peak=False, rms=True, rms_gain=gain, cutoff=cutoff)

    if np.abs(wave).max() < 0.99:
        return gain
    else:
        return chech_gain(wave, fs, gain - 1, cutoff)
