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
