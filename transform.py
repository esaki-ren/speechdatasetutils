from threading import Lock

import numpy as np
from scipy.signal import istft, stft


def phase_augmentation(wave, invert_ratio=0.5, shift_ratio=0.5):
    if np.random.rand() < invert_ratio:
        wave = wave * -1.0

    if np.random.rand() < shift_ratio:
        _, _, Zxx = stft(wave)
        H = np.abs(Zxx)
        P = np.angle(Zxx)
        w = (np.random.rand() * 2 - 1) * np.pi
        P += w
        Zxx = H * np.exp(P * 1j)
        _, wave = istft(Zxx)

    return wave


def spec_augment(spec, t_rate=0.05, f_rate=0.1):
    """
    spec: (time axis * frequency axis)
    """

    with Lock():
        spec = spec.copy()
    tau, nu = spec.shape
    T = int(tau * t_rate)
    F = int(nu * f_rate)

    if T > 0:
        t0 = np.random.randint(tau - T)
        spec[t0:t0+T] = spec[t0:t0+T].mean()

    if F > 0:
        f0 = np.random.randint(nu - F)
        spec[:, f0:f0+F] = spec[:, f0:f0+F].mean()

    return spec
