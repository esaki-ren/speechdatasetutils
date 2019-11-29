
import numpy as np
from scipy import signal


def remove_dc(waveform, fs, numtaps=1025, cutoff=15):
    b = signal.firwin(numtaps, cutoff, pass_zero=False, nyq=fs/2)
    return signal.filtfilt(b, [1], waveform)


def normalize_peak(waveform):
    scale = 0.99 / np.abs(waveform).max()
    return waveform * scale


def normalize_rms(waveform, fs, gain=-11.0):

    nshift = int(fs*0.05)
    pad = waveform.size % nshift
    wave_dc = np.pad(waveform, (0, pad), "constant")
    framed = wave_dc.reshape(-1, nshift)
    rms = np.sqrt((framed**2.0).mean(1))

    idx = int(rms.size*0.95)
    peak = np.sort(rms)[idx]
    peak_db = 20 * np.log10(peak)
    scale_db = gain - peak_db
    scale = 10.0 ** (scale_db / 20.0)

    return waveform * scale


def normalize(waveform, fs, dc_removal=True, peak=True, rms=False, rms_gain=-11.0):
    if dc_removal:
        waveform = remove_dc(waveform, fs)

    if peak:
        ret = normalize_peak(waveform)
    elif rms:
        ret = normalize_rms(waveform, fs, gain=rms_gain)

    return ret
