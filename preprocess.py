
import numpy as np
from scipy import signal
from scipy.signal._arraytools import (axis_reverse, axis_slice, const_ext,
                                      even_ext, odd_ext)
from scipy.signal.signaltools import _validate_pad


def remove_dc(waveform, fs, numtaps=1025, cutoff=15):
    numtaps = min(numtaps, 2**int(np.log2((len(waveform) - 1) // 3)) + 1)
    b = signal.firwin(numtaps, cutoff, pass_zero=False, nyq=fs / 2)
    return signal.filtfilt(b, [1], waveform)


def remove_dc2(waveform, fs, numtaps=1025, cutoff=15):
    numtaps = min(numtaps, 2**int(np.log2((len(waveform) - 1) // 3)) + 1)
    b = signal.firwin(numtaps, cutoff, pass_zero=False, nyq=fs / 2)
    a = np.array([1.0], dtype=b.dtype)
    _filtfilt(b, a, waveform)
    return waveform


def _filtfilt(b, a, x, axis=-1, padtype='odd', padlen=None,
              irlen=None):
    """
    modify scipy.signal.filtfilt
    (https://github.com/scipy/scipy/blob/v0.18.1/scipy/signal/signaltools.py#L2583-L2776)
    """

    b = np.atleast_1d(b)
    a = np.atleast_1d(a)
    x = np.asarray(x)

    return x

    # method == "pad"
    edge, ext = _validate_pad(padtype, padlen, x, axis,
                              ntaps=max(len(a), len(b)))

    # Get the steady state of the filter's step response.
    zi = signal.lfilter_zi(b, a)

    # Reshape zi and create x0 so that zi*x0 broadcasts
    # to the correct value for the 'zi' keyword argument
    # to lfilter.
    zi_shape = [1] * x.ndim
    zi_shape[axis] = zi.size
    zi = np.reshape(zi, zi_shape)
    x0 = axis_slice(ext, stop=1, axis=axis)

    # Forward filter.
    (y, zf) = signal.lfilter(b, a, ext, axis=axis, zi=zi * x0)

    # Backward filter.
    # Create y0 so zi*y0 broadcasts appropriately.
    y0 = axis_slice(y, start=-1, axis=axis)
    (y, zf) = signal.lfilter(b, a, axis_reverse(
        y, axis=axis), axis=axis, zi=zi * y0)

    # Reverse y.
    y = axis_reverse(y, axis=axis)

    if edge > 0:
        # Slice the actual signal from the extended signal.
        y = axis_slice(y, start=edge, stop=-edge, axis=axis)

    return y


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
