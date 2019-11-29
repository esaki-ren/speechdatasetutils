
import numpy as np
from scipy import signal


def remove_dc(waveform, fs, numtaps=1025, cutoff=15):
    b = signal.firwin(numtaps, cutoff, pass_zero=False, nyq=fs/2)
    return signal.filtfilt(b, [1], waveform)


def normalize_peak(waveform):
    scale = 0.99 / np.abs(waveform).max()
    return waveform * scale


def normalize(waveform, fs, dc_removal=True, peak=True, rms=False):
    if dc_removal:
        waveform = remove_dc(waveform, fs)

    if peak:
        ret = normalize_peak(waveform)
    elif rms:
        raise NotImplementedError

    return ret
