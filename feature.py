import os

import numpy as np
import pandas as pd
import pysptk
import pyworld
from librosa import load
from librosa.feature import melspectrogram
from nnmnkwii.preprocessing import preemphasis
from scipy import signal
from scipy.interpolate import PchipInterpolator
from scipy.io import loadmat, savemat, wavfile


def wave2spec(wave, fs, nperseg, frame_period, window, nmels=80, rescaling=True, preemphasis_coef=None, f_min=0, f_max=None, dtype='float32'):
    noverlap = nperseg - (fs * frame_period // 1000)
    assert signal.check_COLA(window, nperseg, noverlap)

    if rescaling:
        wave /= np.max(np.abs(wave))
        wave *= 0.99
    if preemphasis_coef is not None:
        spec_wave = preemphasis(wave, preemphasis_coef)
    else:
        spec_wave = wave
    _, _, Zxx = signal.stft(
        spec_wave, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    pspec = np.abs(Zxx)
    mspec = melspectrogram(
        sr=fs, S=pspec, n_mels=nmels, fmin=f_min, fmax=f_max, power=1.0)
    pspec = pspec.T.astype(dtype)
    mspec = mspec.T.astype(dtype)
    upsample = fs // (1000 // frame_period)
    length = (len(wave) // upsample) * upsample
    wave = wave[:length]
    mspec = mspec[:length // upsample]
    spec = pspec[:length // upsample]

    return wave, spec, mspec, upsample


def wav2world(wave, fs, mcep_order=24, f0_smoothing=-1, ap_smoothing=10, mcep_smoothing=50, frame_period=None, f0_floor=None, f0_ceil=None):
    # setup default values
    wave = wave.astype('float64')

    frame_period = pyworld.default_frame_period if frame_period is None else frame_period
    f0_floor = pyworld.default_f0_floor if f0_floor is None else f0_floor
    f0_ceil = pyworld.default_f0_ceil if f0_ceil is None else f0_ceil
    alpha = pysptk.util.mcepalpha(fs)

    # world
    f0, t = pyworld.harvest(wave, fs,
                            f0_floor=f0_floor,
                            f0_ceil=f0_ceil,
                            frame_period=frame_period)
    sp = pyworld.cheaptrick(wave, f0, t, fs)
    ap = pyworld.d4c(wave, f0, t, fs)

    # extract vuv from ap
    vuv_b = ap[:, 0] < 0.5
    vuv = vuv_b.astype('int')

    # continuous log f0
    idx = np.arange(len(f0))
    vuv_b[0] = vuv_b[-1] = True
    f0[0] = f0[idx[vuv_b]][1]
    f0[-1] = f0[idx[vuv_b]][-2]

    clf0 = np.zeros_like(f0)
    clf0[idx[vuv_b]] = np.log10(f0[idx[vuv_b]])
    clf0[idx[~vuv_b]] = PchipInterpolator(
        idx[vuv_b], clf0[idx[vuv_b]])(idx[~vuv_b])

    if f0_smoothing > 0:
        clf0 = modspec_smoothing(
            clf0, 1000 / frame_period, cut_off=f0_smoothing)

    # continuous coded ap
    cap = pyworld.code_aperiodicity(ap, fs)
    cap[0] = cap[-1] = 1
    cap[idx[~vuv_b]] = PchipInterpolator(
        idx[vuv_b], cap[idx[vuv_b]])(idx[~vuv_b])

    if ap_smoothing > 0:
        cap = modspec_smoothing(cap, 1000 / frame_period, cut_off=ap_smoothing)

    # mcep
    mcep = pysptk.mcep(sp, order=mcep_order, alpha=alpha, itype=4)

    if ap_smoothing > 0:
        mcep = modspec_smoothing(
            mcep, 1000 / frame_period, cut_off=mcep_smoothing)

    fbin = sp.shape[1]
    return mcep, clf0, vuv, cap, sp, fbin, t


def modspec_smoothing(array, fs, cut_off=30, axis=0):
    h = signal.firwin(129, cut_off, nyq=fs // 2)
    return signal.filtfilt(h, 1, array, axis)


def world2wav(clf0, vuv, cap, fs, fbin, mcep=None, sp=None, frame_period=None):

    # setup
    frame_period = pyworld.default_frame_period if frame_period is None else frame_period
    
    clf0 = np.ascontiguousarray(clf0.astype('float64'))
    vuv = np.ascontiguousarray(vuv > 0.5).astype('int')
    cap = np.ascontiguousarray(cap.astype('float64'))
    fft_len = fbin * 2 - 2
    alpha = pysptk.util.mcepalpha(fs)

    # clf0 2 f0
    f0 = 10**clf0 * vuv

    # cap 2 ap
    if cap.ndim != 2:
        cap = np.expand_dims(cap, 1)
    ap = pyworld.decode_aperiodicity(cap, fs, fft_len)

    # mcep 2 sp
    if sp is None:
        if mcep is None:
            raise ValueError

        else:
            mcep = np.ascontiguousarray(mcep)
            sp = pysptk.mgc2sp(mcep, alpha=alpha, fftlen=fft_len)
            sp = np.abs(np.exp(sp)) ** 2
    else:
        sp = np.ascontiguousarray(sp)

    wave = pyworld.synthesize(f0, sp, ap, fs)

    scale = np.abs(wave).max()
    if scale > 0.99:
        wave = wave / scale * 0.99

    return wave
