import numpy as np
from scipy.signal import istft, stft


def phase_augmentation(wave):
    _, _, Zxx = stft(wave)
    H = np.abs(Zxx)
    P = np.angle(Zxx)
    w = np.random.rand() * 4 - 2
    P += w * np.pi
    Zxx = H * np.exp(P * 1j)
    _, x = istft(Zxx)

    return x
