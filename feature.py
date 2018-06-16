import os

import numpy as np
import pandas as pd
import pysptk
import pyworld
from librosa import load
from librosa.feature import melspectrogram
from scipy import signal
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.io import loadmat, savemat, wavfile


def wave2spec(wave, fs, nperseg, frame_period, window, nmels=80, rescaling=True, dtype='float32'):
    noverlap = nperseg - (fs * frame_period // 1000)
    assert signal.check_COLA(window, nperseg, noverlap)

    if rescaling:
        wave /= np.max(np.abs(wave))
        wave *= 0.95
    _, _, Zxx = signal.stft(
        wave, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    pspec = np.abs(Zxx)
    mspec = melspectrogram(
        sr=fs, S=pspec, n_mels=nmels, power=1.0)
    pspec = pspec.T.astype(dtype)
    mspec = mspec.T.astype(dtype)
    upsample = fs // (1000 // frame_period)
    length = (len(wave) // upsample) * upsample
    wave = wave[:length]
    mspec = mspec[:length // upsample]
    spec = pspec[:length // upsample]

    return wave, spec, mspec, upsample


"""
def ap_extract(ap, fs, band):
    
    #extract bap and vuv from ap
    
    # bap
    bap = np.zeros([ap.shape[0], len(band) - 1])
    f = np.linspace(0, fs // 2, num=ap.shape[1])

    for i in range(len(band) - 1):

        index = (f >= band[i]) & (f <= band[i + 1])
        rep = np.sum(index)
        bap[:, i] = np.mean(ap[:, index], axis=1).reshape(-1)

    # vuv
    vuv = ap[:, 0] < 0.5
    return bap, vuv.astype('int')


def wav2world(filepath, f0_floor, f0_ceil, frame_period, apband, fs=16000):
    fs, wav = wavfile.read(filepath)
    wav = wav / 2**15

    # world
    f0, t = pyworld.harvest(wav, fs,
                            f0_floor=f0_floor,
                            f0_ceil=f0_ceil,
                            frame_period=frame_period)
    sp = pyworld.cheaptrick(wav, f0, t, fs)
    ap = pyworld.d4c(wav, f0, t, fs)

    # extract band ap and vuv from ap
    bap, vuv = ap_extract(ap, fs, apband)

    fbin = sp.shape[1]
    return f0, t, bap, vuv, fbin, fs, sp


def wavfile2spec(filepath, fs, window, nperseg, noverlap, n_mels, dtype='float32'):
    y, sr = load(filepath, sr=fs, mono=True, res_type='scipy')
    f, t, Zxx = signal.stft(y, fs=fs, window=window,
                            nperseg=nperseg, noverlap=noverlap)
    espec = np.abs(Zxx)
    mspec = melspectrogram(sr=sr, S=espec, n_mels=n_mels, power=1.0)
    return espec.T.astype(dtype), mspec.T.astype(dtype), t, fs
"""
