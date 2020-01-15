
import numpy as np
from scipy import signal


def remove_dc(waveform, fs, numtaps=1025, cutoff=15):
    numtaps = min(numtaps, 2**int(np.log2((len(waveform)-1)//3)) + 1)
    b = signal.firwin(numtaps, cutoff, pass_zero=False, nyq=fs / 2)
    return signal.filtfilt(b, [1], waveform)


def remove_dc2(waveform, fs, numtaps=1025, cutoff=15):
    # numtaps = min(numtaps, 2**int(np.log2((len(waveform)-1)//3)) + 1)
    # b = signal.firwin(numtaps, cutoff, pass_zero=False, nyq=fs / 2)
    # return signal.filtfilt(b, [1], waveform)
    return waveform


def normalize_peak(waveform):
    scale = 0.99 / np.abs(waveform).max()
    return waveform * scale


def normalize_rms(waveform, fs, gain=-11.0):

    nshift = int(fs * 0.05)
    r = waveform.size % nshift
    if r != 0:
        waveform = np.pad(waveform, (0, nshift - r), "constant")

    framed = waveform.reshape(-1, nshift)
    rms = np.sqrt((framed**2.0).mean(1))

    peak = rms.max()
    peak_db = 20 * np.log10(peak)
    scale_db = gain - peak_db
    scale = 10.0 ** (scale_db / 20.0)

    return waveform * scale


def normalize(
        waveform, fs, dc_removal=True,
        peak=True, rms=False, rms_gain=-11.0, cutoff=15):

    waveform = waveform.copy()

    if dc_removal:
        # waveform = remove_dc(waveform, fs)
        waveform = remove_dc2(waveform, fs)

    if peak:
        waveform = normalize_peak(waveform)
    elif rms:
        waveform = normalize_rms(waveform, fs, gain=rms_gain)

    return waveform
