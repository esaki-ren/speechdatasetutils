import os

import numpy as np
import pandas as pd
import pysptk
import pyworld
from librosa import load
from librosa.feature import melspectrogram
from nnmnkwii.postfilters import merlin_post_filter
from nnmnkwii.preprocessing import preemphasis
from pyreaper import reaper
from scipy import signal
from scipy.interpolate import PchipInterpolator
from scipy.io import loadmat, savemat, wavfile

from .utils import make_stft_args


def wave2spec(wave, fs, frame_period, window, nperseg=None,
              nmels=80, rescaling=True, preemphasis_coef=None,
              f_min=0, f_max=None, dtype='float32', return_t=False):
    stft_kwargs = make_stft_args(
        frame_period, fs, nperseg=nperseg, window=window)

    if rescaling:
        wave /= np.max(np.abs(wave))
        wave *= 0.99
    if preemphasis_coef is not None:
        spec_wave = preemphasis(wave, preemphasis_coef)
    else:
        spec_wave = wave
    _, t, Zxx = signal.stft(
        spec_wave, **stft_kwargs)
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

    if return_t:
        return wave, spec, mspec, upsample, t
    else:
        return wave, spec, mspec, upsample


def wav2world(wave, fs, mcep_order=25, f0_smoothing=30, ap_smoothing=0,
              mcep_smoothing=0, frame_period=None,
              f0_floor=None, f0_ceil=None):
    # setup default values
    wave = wave.astype('float64')

    frame_period = pyworld.default_frame_period if frame_period is None else frame_period
    # f0_floor = pyworld.default_f0_floor if f0_floor is None else f0_floor
    # f0_ceil = pyworld.default_f0_ceil if f0_ceil is None else f0_ceil
    f0_floor = 40.0 if f0_floor is None else f0_floor
    f0_ceil = 500.0 if f0_ceil is None else f0_ceil
    alpha = pysptk.util.mcepalpha(fs)

    # reaper
    _, _, t, f0, _ = reaper(
        (wave * (2**15 - 1)).astype("int16"), fs,
        minf0=f0_floor, maxf0=f0_ceil,
        do_high_pass=True, do_hilbert_transform=True,
        inter_pulse=0.01, frame_period=frame_period/1000,
        unvoiced_cost=0.9)
    t = t.astype("float64")
    f0 = f0.astype("float64")

    # world
    """
    f0, t = pyworld.harvest(wave, fs,
                            f0_floor=f0_floor,
                            f0_ceil=f0_ceil,
                            frame_period=frame_period)
    """
    sp = pyworld.cheaptrick(wave, f0, t, fs)
    ap = pyworld.d4c(wave, f0, t, fs, threshold=0.85)

    # extract vuv from ap and f0
    vuv = (ap[:, 0] < 0.5).astype('int')
    vuv_b=(ap[:, 0] < 0.5)
    vuv_b = f0*vuv > 1
    vuv = vuv_b.astype('int')

    # continuous log f0
    idx = np.arange(len(f0))
    if not vuv_b[0]:
        f0[0] = f0[idx[vuv_b]].mean()
        vuv_b[0] = True
    if not vuv_b[-1]:
        f0[-1] = f0[idx[vuv_b]].mean()
        vuv_b[-1] = True

    clf0 = np.zeros_like(f0)
    clf0[idx[vuv_b]] = np.log(np.clip(f0[idx[vuv_b]], f0_floor/2, f0_ceil*2))
    clf0[idx[~vuv_b]] = PchipInterpolator(
        idx[vuv_b], clf0[idx[vuv_b]])(idx[~vuv_b])

    if f0_smoothing > 0:
        clf0 = modspec_smoothing(
            clf0, 1000 / frame_period, cut_off=f0_smoothing)

    # continuous coded ap
    cap = pyworld.code_aperiodicity(ap, fs)

    if ap_smoothing > 0:
        cap = modspec_smoothing(cap, 1000 / frame_period, cut_off=ap_smoothing)

    # mcep
    print(sp.shape)
    mcep = pysptk.mcep(sp, order=mcep_order, alpha=alpha, itype=4)
    print(mcep.shape)

    if mcep_smoothing > 0:
        mcep = modspec_smoothing(
            mcep, 1000 / frame_period, cut_off=mcep_smoothing)

    fbin = sp.shape[1]
    return mcep, clf0, vuv, cap, sp, fbin, t


def f0_extract(wave, fs, frame_period=None, f0_floor=None, f0_ceil=None):
    # setup default values
    wave = wave.astype('float64')

    frame_period = pyworld.default_frame_period if frame_period is None else frame_period
    f0_floor = pyworld.default_f0_floor if f0_floor is None else f0_floor
    f0_ceil = pyworld.default_f0_ceil if f0_ceil is None else f0_ceil

    # world
    f0, t = pyworld.harvest(wave, fs,
                            f0_floor=f0_floor,
                            f0_ceil=f0_ceil,
                            frame_period=frame_period)
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
    clf0[idx[vuv_b]] = np.log(np.clip(f0[idx[vuv_b]], f0_floor/2, f0_ceil*2))
    clf0[idx[~vuv_b]] = PchipInterpolator(
        idx[vuv_b], clf0[idx[vuv_b]])(idx[~vuv_b])

    return clf0, vuv, t


def modspec_smoothing(array, fs, cut_off=30, axis=0, fbin=11):
    if cut_off >= fs / 2:
        return array
    h = signal.firwin(fbin, cut_off, nyq=fs // 2)
    return signal.filtfilt(h, 1, array, axis)


def world2wav(clf0, vuv, cap, fs, fbin, mcep=None, sp=None, frame_period=None, mcep_postfilter=False):

    # setup
    frame_period = pyworld.default_frame_period if frame_period is None else frame_period

    clf0 = np.ascontiguousarray(clf0.astype('float64'))
    vuv = np.ascontiguousarray(vuv > 0.5).astype('int')
    cap = np.ascontiguousarray(cap.astype('float64'))
    fft_len = fbin * 2 - 2
    alpha = pysptk.util.mcepalpha(fs)

    # clf0 2 f0
    f0 = np.squeeze(np.exp(clf0)) * np.squeeze(vuv)

    # cap 2 ap
    if cap.ndim != 2:
        cap = np.expand_dims(cap, 1)
    ap = pyworld.decode_aperiodicity(cap, fs, fft_len)

    # mcep 2 sp
    if sp is None:
        if mcep is None:
            raise ValueError

        else:
            mcep = np.ascontiguousarray(mcep.astype('float64'))
            if mcep_postfilter:
                mcep = merlin_post_filter(mcep, alpha)
            sp = pysptk.mgc2sp(mcep, alpha=alpha, fftlen=fft_len)
            sp = np.abs(np.exp(sp)) ** 2
    else:
        sp = np.ascontiguousarray(sp)

    wave = pyworld.synthesize(f0, sp, ap, fs, frame_period=frame_period)

    scale = np.abs(wave).max()
    if scale > 0.99:
        wave = wave / scale * 0.99

    return wave
